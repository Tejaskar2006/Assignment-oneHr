# DECISIONS.md

1. **Indexes.** Unique `emp_code` rejects a duplicate employee create. Unique `(emp_code, date)` does the same for punch-in. `date_emp` covers the attendance sort, `emp_punch_in` finds the punch to close, and `status_date_emp` covers a status filter. `department_joined` and `joined_on` cover headcount. I skipped a lone `date` index because `date_emp` already serves that range and the sort.

2. **Punch-in race.** Both requests compute the same date and insert. The unique index commits one insert. The other raises a duplicate-key error and returns 409. There is no read before the write.

3. **Ties.** Rows are ordered by total late minutes, then `emp_code`. Rank is one plus the people strictly ahead, so a tie shares a rank and the next rank is skipped. `limit` returns every row with rank at most `limit`: a tie on the cutoff adds rows, and a rank past it is dropped.

4. **Headcount.** The summary starts from employees with `joined_on` on or before month-end, then left-joins that month's logs. Someone with no logs still counts. A department with nobody employed yet is omitted.

5. **At 100x** I would keep a daily department rollup, updated on each punch and correction, so the trend does not reread every log.
