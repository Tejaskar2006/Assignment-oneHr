"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded; existing environment variables win)
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Annotated, Any, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.functional_validators import BeforeValidator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))
PRESENCE = ("PRESENT", "WFH", "ON_DUTY")
ABSENCE = ("ABSENT", "LEAVE")
EPOCH_MIN = 100_000_000_000
EPOCH_MAX = 4_102_444_800_000
TRACKED = (
    "status",
    "punch_in",
    "punch_out",
    "work_hours",
    "late_minutes",
    "overtime_minutes",
    "half_day",
)

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=timezone.utc,
    serverSelectionTimeoutMS=8000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]


# --------------------------------------------------------------------------- #
# Time, rounding, serialization
# --------------------------------------------------------------------------- #
def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def as_ist(dt: datetime) -> datetime:
    return as_utc(dt).astimezone(IST)


def ms_to_dt(ms: int) -> datetime:
    """Epoch milliseconds, truncated to a whole second, as an aware UTC datetime."""
    return datetime.fromtimestamp(int(ms) // 1000, tz=timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_epoch_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(as_utc(value).timestamp() * 1000)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return parse_hhmm(shift_end) <= parse_hhmm(shift_start)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1. IST calendar date, previous day when an overnight punch is before shift end."""
    ist = as_ist(punch_in)
    day = ist.date()
    if is_overnight(shift_start, shift_end):
        end_h, end_m = parse_hhmm(shift_end)
        if (ist.hour, ist.minute, ist.second) < (end_h, end_m, 0):
            day -= timedelta(days=1)
    return day.isoformat()


def ist_clock(date_str: str, hhmm: str) -> datetime:
    hour, minute = parse_hhmm(hhmm)
    year, month, day = (int(part) for part in date_str.split("-"))
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def shift_end_at(date_str: str, shift_start: str, shift_end: str) -> datetime:
    end = ist_clock(date_str, shift_end)
    if is_overnight(shift_start, shift_end):
        end += timedelta(days=1)
    return end


def late_minutes(punch_in: datetime, date_str: str, shift_start: str) -> int:
    """R2. Whole minutes since shift start, and only after a strict 10:00 grace."""
    delta = (as_utc(punch_in) - as_utc(ist_clock(date_str, shift_start))).total_seconds()
    if delta <= 10 * 60:
        return 0
    return int(delta // 60)


def overtime_minutes(punch_out: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R3. Whole minutes after shift end, counted only when that is at least 30."""
    delta = (as_utc(punch_out) - as_utc(shift_end_at(date_str, shift_start, shift_end))).total_seconds()
    minutes = int(delta // 60)
    if minutes < 30:
        return 0
    return minutes


def work_hours_amount(punch_in: datetime, punch_out: datetime) -> Decimal:
    """R4. Elapsed hours, 2 decimals, half-up."""
    seconds = int((as_utc(punch_out) - as_utc(punch_in)).total_seconds())
    return (Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def work_hours(punch_in: datetime, punch_out: datetime) -> float:
    return float(work_hours_amount(punch_in, punch_out))


def is_half_day(hours: Decimal) -> bool:
    """R5. Compared against the already-rounded work hours."""
    return hours < Decimal("4.50")


def round_half_up(value: Decimal, places: int) -> float:
    quant = Decimal("1").scaleb(-places)
    return float(value.quantize(quant, rounding=ROUND_HALF_UP))


def parse_date(value: str, field: str, where: str = "query") -> str:
    try:
        date.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": [where, field], "msg": "invalid date", "type": "value_error"}],
        ) from None
    return value


def month_bounds(month: str) -> tuple[str, str]:
    year, mon = (int(part) for part in month.split("-"))
    start = date(year, mon, 1)
    if mon == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, mon + 1, 1) - timedelta(days=1)
    return start.isoformat(), end.isoformat()


def working_days(month: str, joined_on: str) -> int:
    """R7. Mon-Fri from joined_on through the end of the month."""
    start_s, end_s = month_bounds(month)
    start = max(date.fromisoformat(start_s), date.fromisoformat(joined_on))
    end = date.fromisoformat(end_s)
    if start > end:
        return 0
    count = 0
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


def hours_value(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def serialize_history(history: Optional[list]) -> list:
    entries = []
    for entry in history or []:
        changes = {}
        for key, change in (entry.get("changes") or {}).items():
            old, new = change.get("from"), change.get("to")
            if key in ("punch_in", "punch_out"):
                old = to_epoch_ms(old) if isinstance(old, datetime) else old
                new = to_epoch_ms(new) if isinstance(new, datetime) else new
            changes[key] = {"from": old, "to": new}
        at = entry.get("at")
        entries.append(
            {
                "at": to_epoch_ms(at) if isinstance(at, datetime) else at,
                "by": entry.get("by"),
                "reason": entry.get("reason"),
                "changes": changes,
            }
        )
    return entries


def serialize_employee(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": to_epoch_ms(doc.get("created_at")),
    }


def serialize_attendance(doc: dict) -> dict:
    work = doc.get("work_hours")
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_epoch_ms(doc.get("punch_in")),
        "punch_out": to_epoch_ms(doc.get("punch_out")),
        "work_hours": None if work is None else float(work),
        "late_minutes": int(doc.get("late_minutes") or 0),
        "overtime_minutes": int(doc.get("overtime_minutes") or 0),
        "half_day": bool(doc.get("half_day")),
        "history": serialize_history(doc.get("history")),
    }


def comparable(doc: dict) -> dict:
    return {
        "status": doc.get("status"),
        "punch_in": to_epoch_ms(doc.get("punch_in")),
        "punch_out": to_epoch_ms(doc.get("punch_out")),
        "work_hours": hours_value(doc.get("work_hours")),
        "late_minutes": int(doc.get("late_minutes") or 0),
        "overtime_minutes": int(doc.get("overtime_minutes") or 0),
        "half_day": bool(doc.get("half_day")),
    }


def stored_change_value(key: str, value: Any) -> Any:
    if key in ("punch_in", "punch_out"):
        return None if value is None else ms_to_dt(value)
    if key == "work_hours":
        return None if value is None else float(value)
    return value


def span_ok(punch_in: datetime, punch_out: datetime) -> bool:
    delta = (as_utc(punch_out) - as_utc(punch_in)).total_seconds()
    return 0 < delta <= 24 * 3600


def derived_for(status: str, punch_in: Optional[datetime], punch_out: Optional[datetime], day: str, employee: dict) -> dict:
    if status in ABSENCE:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    late = late_minutes(punch_in, day, employee["shift_start"]) if punch_in is not None else 0
    if punch_out is None or punch_in is None:
        return {"work_hours": None, "late_minutes": late, "overtime_minutes": 0, "half_day": False}
    hours = work_hours_amount(punch_in, punch_out)
    return {
        "work_hours": float(hours),
        "late_minutes": late,
        "overtime_minutes": overtime_minutes(punch_out, day, employee["shift_start"], employee["shift_end"]),
        "half_day": is_half_day(hours),
    }


def weekday_cond(date_expr: str) -> dict:
    return {
        "$in": [
            {"$dayOfWeek": {"$dateFromString": {"dateString": date_expr, "format": "%Y-%m-%d", "timezone": "UTC"}}},
            [2, 3, 4, 5, 6],
        ]
    }


def present_contribution(status_expr: str, half_expr: str, date_expr: str) -> dict:
    return {
        "$cond": [
            {
                "$and": [
                    {"$in": [status_expr, list(PRESENCE)]},
                    weekday_cond(date_expr),
                ]
            },
            {"$cond": [{"$eq": [half_expr, True]}, 0.5, 1]},
            0,
        ]
    }


# --------------------------------------------------------------------------- #
# Indexes
# --------------------------------------------------------------------------- #
def ensure_indexes() -> None:
    db.employees.create_index([("emp_code", 1)], unique=True, name="emp_code_unique")
    db.employees.create_index([("department", 1), ("joined_on", 1)], name="department_joined")
    db.employees.create_index([("joined_on", 1)], name="joined_on")
    db.attendance_logs.create_index(
        [("emp_code", 1), ("date", 1)], unique=True, name="emp_code_date_unique"
    )
    db.attendance_logs.create_index([("date", -1), ("emp_code", 1)], name="date_emp")
    db.attendance_logs.create_index([("emp_code", 1), ("punch_in", -1)], name="emp_punch_in")
    db.attendance_logs.create_index(
        [("status", 1), ("date", -1), ("emp_code", 1)], name="status_date_emp"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _epoch(value: Any) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError("must be integer epoch milliseconds")
    if value < EPOCH_MIN or value > EPOCH_MAX:
        raise ValueError("epoch milliseconds out of range")
    return value


EpochMs = Annotated[int, BeforeValidator(_epoch)]


class PresenceStatus(str, Enum):
    PRESENT = "PRESENT"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class Status(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    LEAVE = "LEAVE"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def check_shift_and_date(self) -> "EmployeeIn":
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        try:
            date.fromisoformat(self.joined_on)
        except ValueError as exc:
            raise ValueError("joined_on must be a real YYYY-MM-DD date") from exc
        return self


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = PresenceStatus.PRESENT


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# Query builders (shared with /admin/explain)
# --------------------------------------------------------------------------- #
def attendance_filter(
    emp_code: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    status: Optional[str],
) -> dict:
    if date_from is not None:
        date_from = parse_date(date_from, "date_from")
    if date_to is not None:
        date_to = parse_date(date_to, "date_to")
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["query", "date_from"], "msg": "date_from must be on or before date_to", "type": "value_error"}],
        )
    query: dict[str, Any] = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    if date_from is not None or date_to is not None:
        bounds: dict[str, str] = {}
        if date_from is not None:
            bounds["$gte"] = date_from
        if date_to is not None:
            bounds["$lte"] = date_to
        query["date"] = bounds
    if status is not None:
        query["status"] = status
    return query


def attendance_hint(query: dict) -> str:
    if "emp_code" in query:
        return "emp_code_date_unique"
    if "status" in query:
        return "status_date_emp"
    return "date_emp"


def attendance_find_command(query: dict, page: int, page_size: int) -> dict:
    return {
        "find": "attendance_logs",
        "filter": query,
        "sort": {"date": -1, "emp_code": 1},
        "skip": (page - 1) * page_size,
        "limit": page_size,
        "hint": attendance_hint(query),
    }


def employee_monthly_pipeline(emp_code: str, month: str) -> list:
    start, end = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lte": end}}},
        {
            "$group": {
                "_id": None,
                "present_days": {"$sum": present_contribution("$status", "$half_day", "$date")},
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {
                    "$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]


def department_summary_pipeline(month: str, department: Optional[str]) -> tuple[list, dict]:
    start, end = month_bounds(month)
    match: dict[str, Any] = {"joined_on": {"$lte": end}}
    if department is not None:
        match["department"] = department
        hint = "department_joined"
    else:
        hint = "joined_on"
    pipeline = [
        {"$match": match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "logs",
            }
        },
        {
            "$set": {
                "logs": {
                    "$filter": {
                        "input": "$logs",
                        "as": "log",
                        "cond": {
                            "$and": [
                                {"$gte": ["$$log.date", start]},
                                {"$lte": ["$$log.date", end]},
                            ]
                        },
                    }
                }
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": "$department",
                "people": {"$addToSet": "$emp_code"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {"$ifNull": ["$logs.date", False]},
                            present_contribution("$logs.status", "$logs.half_day", "$logs.date"),
                            0,
                        ]
                    }
                },
                "wh_sum": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": [{"$ifNull": ["$logs.status", ""]}, list(PRESENCE)]},
                                    {"$ne": [{"$type": "$logs.work_hours"}, "missing"]},
                                    {"$ne": ["$logs.work_hours", None]},
                                ]
                            },
                            "$logs.work_hours",
                            0,
                        ]
                    }
                },
                "wh_n": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": [{"$ifNull": ["$logs.status", ""]}, list(PRESENCE)]},
                                    {"$ne": [{"$type": "$logs.work_hours"}, "missing"]},
                                    {"$ne": ["$logs.work_hours", None]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]},
                            1,
                            0,
                        ]
                    }
                },
                "total_late_minutes": {"$sum": {"$ifNull": ["$logs.late_minutes", 0]}},
                "leave_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}},
                "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    return pipeline, hint


def late_leaderboard_pipeline(month: str, department: Optional[str]) -> list:
    start, end = month_bounds(month)
    pipeline: list[dict] = [
        {"$match": {"date": {"$gte": start, "$lte": end}}},
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "late_count": {
                    "$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}
                },
            }
        },
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})
    pipeline.append({"$sort": {"total_late_minutes": -1, "_id": 1}})
    return pipeline


def trend_inputs(department: str, date_from: str, date_to: str) -> tuple[list[str], list[str]]:
    date_from = parse_date(date_from, "from")
    date_to = parse_date(date_to, "to")
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    if end < start:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["query", "to"], "msg": "to must be on or after from", "type": "value_error"}],
        )
    if (end - start).days + 1 > 92:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["query", "to"], "msg": "range exceeds 92 days", "type": "value_error"}],
        )
    employees = list(db.employees.find({"department": department}, {"_id": 0, "emp_code": 1, "joined_on": 1}))
    return [row["emp_code"] for row in employees], [row["joined_on"] for row in employees]


def utc_midnight(date_str: str) -> datetime:
    year, month, day = (int(part) for part in date_str.split("-"))
    return datetime(year, month, day, tzinfo=timezone.utc)


def department_trend_pipeline(codes: list[str], joined: list[str], date_from: str, date_to: str) -> list:
    # An empty $in can collapse to a collection scan. A code that cannot exist keeps the index plan.
    match_codes = codes or ["__none__"]
    return [
        {"$match": {"emp_code": {"$in": match_codes}, "date": {"$gte": date_from, "$lte": date_to}}},
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", list(PRESENCE)]},
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]},
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}
                },
            }
        },
        {
            "$unionWith": {
                "pipeline": [
                    {
                        "$documents": [
                            {"_id": date_from, "present_count": 0, "late_count": 0},
                            {"_id": date_to, "present_count": 0, "late_count": 0},
                        ]
                    }
                ]
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "present_count": {"$sum": "$present_count"},
                "late_count": {"$sum": "$late_count"},
            }
        },
        {
            "$addFields": {
                "dateObj": {
                    "$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d", "timezone": "UTC"}
                }
            }
        },
        {
            "$densify": {
                "field": "dateObj",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [utc_midnight(date_from), utc_midnight(date_to)],
                },
            }
        },
        {
            "$addFields": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$dateObj", "timezone": "UTC"}},
                "present_count": {"$ifNull": ["$present_count", 0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
                "dow": {"$dayOfWeek": "$dateObj"},
            }
        },
        {
            "$addFields": {
                "is_working_day": {"$and": [{"$gte": ["$dow", 2]}, {"$lte": ["$dow", 6]}]},
                "headcount": {
                    "$size": {
                        "$filter": {
                            "input": joined,
                            "as": "joined_on",
                            "cond": {"$lte": ["$$joined_on", "$date"]},
                        }
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
            }
        },
        {"$sort": {"date": 1}},
    ]


def with_rates(rows: list[dict]) -> list[dict]:
    rates: list[Optional[float]] = []
    shaped = []
    for row in rows:
        present = row.get("present_count") or 0
        headcount = int(row.get("headcount") or 0)
        working = bool(row.get("is_working_day"))
        if working and headcount > 0:
            rate = round_half_up(Decimal(str(present)) / Decimal(headcount), 4)
        else:
            rate = None
        rates.append(rate)
        shaped.append(
            {
                "date": row["date"],
                "is_working_day": working,
                "headcount": headcount,
                "present_count": float(present) if not float(present).is_integer() else int(present),
                "late_count": int(row.get("late_count") or 0),
                "attendance_rate": rate,
            }
        )
    for index, row in enumerate(shaped):
        window = [rate for rate in rates[max(0, index - 6) : index + 1] if rate is not None]
        if not window:
            row["moving_avg_7d"] = None
        else:
            average = sum((Decimal(str(rate)) for rate in window), Decimal(0)) / Decimal(len(window))
            row["moving_avg_7d"] = round_half_up(average, 4)
    return shaped


def number_or_int(value: Any) -> float | int:
    number = float(value or 0)
    if number.is_integer():
        return int(number)
    return number


def explain_command(command: dict) -> dict:
    raw = db.command({"explain": command, "verbosity": "executionStats"})
    # Extended JSON so BSON dates and longs survive the response unchanged.
    return json.loads(json_util.dumps(raw))


def require(value: Optional[str], field: str) -> str:
    if value is None or value == "":
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["query", field], "msg": "Field required", "type": "missing"}],
        )
    return value


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_utc()
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists") from None
    return serialize_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query: dict[str, Any] = {}
    if department is not None:
        query["department"] = department
    total = db.employees.count_documents(query)
    cursor = (
        db.employees.find(query, {"_id": 0})
        .sort([("emp_code", 1)])
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "items": [serialize_employee(doc) for doc in cursor],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    employee = db.employees.find_one({"emp_code": body.emp_code}, {"_id": 0})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")
    punched = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    day = attendance_date(punched, employee["shift_start"], employee["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status.value,
        "punch_in": punched,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_minutes(punched, day, employee["shift_start"]),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="already punched in for this date") from None
    return serialize_attendance(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    employee = db.employees.find_one({"emp_code": body.emp_code}, {"_id": 0})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")
    punched = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$ne": None, "$lte": punched}},
        sort=[("punch_in", -1)],
        hint="emp_punch_in",
    )
    if record is None:
        raise HTTPException(status_code=404, detail="no punch-in found")
    if record.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="already punched out")
    punch_in_at = record["punch_in"]
    if not span_ok(punch_in_at, punched):
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["body", "punched_at"], "msg": "punch-out must be after punch-in and within 24 hours", "type": "value_error"}],
        )
    derived = derived_for(record["status"], punch_in_at, punched, record["date"], employee)
    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {"$set": {"punch_out": punched, **derived}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="already punched out")
    return serialize_attendance(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query = attendance_filter(emp_code, date_from, date_to, status.value if status else None)
    total = db.attendance_logs.count_documents(query, hint=attendance_hint(query))
    cursor = (
        db.attendance_logs.find(query)
        .sort([("date", -1), ("emp_code", 1)])
        .skip((page - 1) * page_size)
        .limit(page_size)
        .hint(attendance_hint(query))
    )
    return {
        "items": [serialize_attendance(doc) for doc in cursor],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    day = parse_date(date, "date", where="path")
    employee = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": day})
    if record is None:
        raise HTTPException(status_code=404, detail="attendance record not found")

    sent = body.model_fields_set
    if "status" in sent and body.status is None:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["body", "status"], "msg": "invalid status", "type": "value_error"}],
        )
    status = body.status.value if "status" in sent else record["status"]
    punch_in_at = record.get("punch_in")
    punch_out_at = record.get("punch_out")

    if status in ABSENCE:
        if "punch_in" in sent or "punch_out" in sent:
            raise HTTPException(
                status_code=422,
                detail=[{"loc": ["body", "status"], "msg": "ABSENT or LEAVE cannot include punch times", "type": "value_error"}],
            )
        punch_in_at = None
        punch_out_at = None
    else:
        if "punch_in" in sent:
            if body.punch_in is None:
                raise HTTPException(
                    status_code=422,
                    detail=[{"loc": ["body", "punch_in"], "msg": "must be integer epoch milliseconds", "type": "value_error"}],
                )
            punch_in_at = ms_to_dt(body.punch_in)
        if "punch_out" in sent:
            if body.punch_out is None:
                raise HTTPException(
                    status_code=422,
                    detail=[{"loc": ["body", "punch_out"], "msg": "must be integer epoch milliseconds", "type": "value_error"}],
                )
            punch_out_at = ms_to_dt(body.punch_out)
        if punch_in_at is None:
            raise HTTPException(
                status_code=422,
                detail=[{"loc": ["body", "punch_in"], "msg": "a presence status requires punch_in", "type": "value_error"}],
            )
        if attendance_date(punch_in_at, employee["shift_start"], employee["shift_end"]) != day:
            raise HTTPException(
                status_code=422,
                detail=[{"loc": ["body", "punch_in"], "msg": "punch_in must stay on the attendance date", "type": "value_error"}],
            )
        if punch_out_at is not None and not span_ok(punch_in_at, punch_out_at):
            raise HTTPException(
                status_code=422,
                detail=[{"loc": ["body", "punch_out"], "msg": "punch_out must be after punch_in and within 24 hours", "type": "value_error"}],
            )

    derived = derived_for(status, punch_in_at, punch_out_at, day, employee)
    new_doc = {
        "status": status,
        "punch_in": punch_in_at,
        "punch_out": punch_out_at,
        **derived,
        "history": record.get("history") or [],
    }
    before = comparable(record)
    after = comparable(new_doc)
    changes = {}
    for key in TRACKED:
        if before[key] != after[key]:
            changes[key] = {
                "from": stored_change_value(key, before[key]),
                "to": stored_change_value(key, after[key]),
            }
    if not changes:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["body"], "msg": "request changes nothing", "type": "value_error"}],
        )

    history = list(record.get("history") or [])
    entry = {
        "at": now_utc(),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }
    history_filter: Any
    if history:
        history_filter = history
    else:
        history_filter = {"$in": [None, []]}
    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "history": history_filter},
        {
            "$set": {
                "status": status,
                "punch_in": punch_in_at,
                "punch_out": punch_out_at,
                **derived,
                "history": history + [entry],
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="attendance record was updated concurrently")
    return serialize_attendance(updated)


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    employee = db.employees.find_one({"emp_code": emp_code}, {"_id": 0, "joined_on": 1})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")
    pipeline = employee_monthly_pipeline(emp_code, month)
    rows = list(db.attendance_logs.aggregate(pipeline, hint="emp_code_date_unique"))
    stats = rows[0] if rows else {}
    days = working_days(month, employee["joined_on"])
    present = stats.get("present_days") or 0
    present_out = number_or_int(present)
    if days == 0:
        pct = None
    else:
        pct = round_half_up(Decimal(str(present)) / Decimal(days) * Decimal(100), 2)
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": days,
        "present_days": present_out,
        "leave_days": int(stats.get("leave_days") or 0),
        "late_count": int(stats.get("late_count") or 0),
        "total_late_minutes": int(stats.get("total_late_minutes") or 0),
        "total_overtime_minutes": int(stats.get("total_overtime_minutes") or 0),
        "attendance_pct": pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    pipeline, hint = department_summary_pipeline(month, department)
    rows = db.employees.aggregate(pipeline, hint=hint)
    items = []
    for row in rows:
        headcount = len(row.get("people") or [])
        if headcount == 0:
            continue
        count = int(row.get("wh_n") or 0)
        if count == 0:
            average = None
        else:
            average = round_half_up(Decimal(str(row.get("wh_sum") or 0)) / Decimal(count), 2)
        items.append(
            {
                "department": row["_id"],
                "headcount": headcount,
                "present_days": number_or_int(row.get("present_days")),
                "avg_work_hours": average,
                "late_count": int(row.get("late_count") or 0),
                "total_late_minutes": int(row.get("total_late_minutes") or 0),
                "leave_count": int(row.get("leave_count") or 0),
                "on_duty_count": int(row.get("on_duty_count") or 0),
            }
        )
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    pipeline = late_leaderboard_pipeline(month, department)
    rows = list(
        db.attendance_logs.aggregate(pipeline, hint="date_emp")
    )
    items = []
    rank = 0
    previous: Optional[int] = None
    for index, row in enumerate(rows):
        total = int(row["total_late_minutes"])
        if previous is None or total < previous:
            rank = index + 1
            previous = total
        if rank > limit:
            break
        items.append(
            {
                "rank": rank,
                "emp_code": row["_id"],
                "name": row["emp"]["name"],
                "department": row["emp"]["department"],
                "total_late_minutes": total,
                "late_count": int(row["late_count"]),
            }
        )
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: str = Query(..., alias="from"),
    date_to: str = Query(..., alias="to"),
):
    codes, joined = trend_inputs(department, date_from, date_to)
    if not codes:
        raise HTTPException(status_code=404, detail="department not found")
    pipeline = department_trend_pipeline(codes, joined, date_from, date_to)
    rows = list(
        db.attendance_logs.aggregate(pipeline, hint="emp_code_date_unique")
    )
    return {"department": department, "items": with_rates(rows)}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    trend_from: Optional[str] = Query(None, alias="from"),
    trend_to: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    allowed = {
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    }
    if endpoint not in allowed:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["path", "endpoint"], "msg": "unknown endpoint", "type": "literal_error"}],
        )
    if endpoint == "attendance_list":
        query = attendance_filter(emp_code, date_from, date_to, status.value if status else None)
        command = attendance_find_command(query, page, page_size)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_command(command)}
    if endpoint == "employee_monthly":
        require(emp_code, "emp_code")
        month = require(month, "month")
        command = {
            "aggregate": "attendance_logs",
            "pipeline": employee_monthly_pipeline(emp_code, month),
            "hint": "emp_code_date_unique",
            "cursor": {},
        }
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_command(command)}
    if endpoint == "department_summary":
        month = require(month, "month")
        pipeline, hint = department_summary_pipeline(month, department)
        command = {
            "aggregate": "employees",
            "pipeline": pipeline,
            "hint": hint,
            "cursor": {},
        }
        return {"endpoint": endpoint, "collection": "employees", "explain": explain_command(command)}
    if endpoint == "late_leaderboard":
        month = require(month, "month")
        command = {
            "aggregate": "attendance_logs",
            "pipeline": late_leaderboard_pipeline(month, department),
            "hint": "date_emp",
            "cursor": {},
        }
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_command(command)}
    trend_from = require(trend_from, "from")
    trend_to = require(trend_to, "to")
    require(department, "department")
    codes, joined = trend_inputs(department, trend_from, trend_to)
    command = {
        "aggregate": "attendance_logs",
        "pipeline": department_trend_pipeline(codes, joined, trend_from, trend_to),
        "hint": "emp_code_date_unique",
        "cursor": {},
    }
    return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_command(command)}
