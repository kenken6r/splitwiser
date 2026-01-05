# db.py (Supabase Postgres version)

import hashlib
import secrets
from datetime import date
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st

import psycopg
from psycopg.rows import dict_row
from psycopg import errors as pg_errors


# ------------------------
# ID helpers
# ------------------------
def new_page_id() -> str:
    """
    8 chars, URL safe, easy to read.
    Uses [a z 0 9] only.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))


def _norm_page_id(page_id) -> str:
    if page_id is None:
        return ""
    return str(page_id).strip()


def _db_url() -> str:
    try:
        return st.secrets["postgres"]["url"]
    except Exception:
        raise RuntimeError("Missing secrets: set [postgres].url in .streamlit/secrets.toml or Streamlit Cloud Secrets")


def get_conn() -> psycopg.Connection:
    # Open a fresh connection each call
    return psycopg.connect(_db_url(), row_factory=dict_row)


def init_db() -> None:
    """
    You already created tables in Supabase SQL Editor.
    Keep this as a light connectivity check.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("select 1 as ok;")
            _ = cur.fetchone()
    finally:
        conn.close()


# ------------------------
# Password helpers
# ------------------------
def _hash_password(password: str, salt_hex: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        200_000,
    )
    return dk.hex()


def _make_password_record(password: str) -> Tuple[str, str]:
    salt_hex = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt_hex)
    return salt_hex, pw_hash


def verify_page_password(page_id, password: str) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    password = (password or "").strip()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select password_salt, password_hash
                from pages
                where id = %s and is_deleted = false
                """,
                (page_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return False, "Page not found"

    salt = row["password_salt"]
    pw_hash = row["password_hash"]

    if salt is None or pw_hash is None:
        return True, "No password"

    if not password:
        return False, "Password required"

    ok = _hash_password(password, salt) == pw_hash
    return (True, "OK") if ok else (False, "Wrong password")


# ------------------------
# Page ops
# ------------------------
def create_page(name: str, password: str) -> Tuple[bool, str]:
    name = (name or "").strip()
    password = (password or "").strip()
    if not name:
        return False, "Page name is empty"

    salt = None
    pw_hash = None
    if password:
        salt, pw_hash = _make_password_record(password)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # id collision is extremely unlikely, but retry anyway
            for _ in range(10):
                pid = new_page_id()
                try:
                    cur.execute(
                        """
                        insert into pages (id, name, password_salt, password_hash)
                        values (%s, %s, %s, %s)
                        """,
                        (pid, name, salt, pw_hash),
                    )
                    conn.commit()
                    return True, "Created"
                except pg_errors.UniqueViolation:
                    conn.rollback()
                    # Could be duplicate name or duplicate id
                    # If name already exists, stop early
                    cur.execute("select 1 from pages where name = %s limit 1", (name,))
                    if cur.fetchone():
                        return False, "That page name already exists"
                    # Otherwise, treat as id collision and retry
                    continue

            return False, "Failed to create page id. Try again."
    finally:
        conn.close()


def list_pages(include_deleted: bool = False) -> List[Dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if include_deleted:
                cur.execute(
                    """
                    select id, name, password_hash, is_deleted, deleted_at
                    from pages
                    order by lower(name)
                    """
                )
            else:
                cur.execute(
                    """
                    select id, name, password_hash
                    from pages
                    where is_deleted = false
                    order by lower(name)
                    """
                )
            return cur.fetchall()
    finally:
        conn.close()


def get_page(page_id) -> Optional[Dict]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, name, password_hash
                from pages
                where id = %s and is_deleted = false
                """,
                (page_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


# ------------------------
# Member ops (page scoped)
# ------------------------
def add_member(page_id, name: str) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    name = (name or "").strip()
    if not name:
        return False, "Name is empty"

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "insert into members (page_id, name) values (%s, %s)",
                    (page_id, name),
                )
                conn.commit()
                return True, "Added"
            except pg_errors.UniqueViolation:
                conn.rollback()
                return False, "That name already exists"
    finally:
        conn.close()


def get_members(page_id, include_deleted: bool = False) -> List[Dict]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if include_deleted:
                cur.execute(
                    """
                    select id, name, is_deleted, deleted_at
                    from members
                    where page_id = %s
                    order by lower(name)
                    """,
                    (page_id,),
                )
            else:
                cur.execute(
                    """
                    select id, name
                    from members
                    where page_id = %s and is_deleted = false
                    order by lower(name)
                    """,
                    (page_id,),
                )
            return cur.fetchall()
    finally:
        conn.close()


def member_usage_count(page_id, member_id: int) -> int:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) as c
                from expenses
                where page_id = %s and paid_by_member_id = %s and is_deleted = false
                """,
                (page_id, int(member_id)),
            )
            c1 = int(cur.fetchone()["c"])

            cur.execute(
                """
                select count(*) as c
                from expense_shares s
                join expenses e on e.id = s.expense_id
                where e.page_id = %s
                  and s.member_id = %s
                  and s.is_deleted = false
                  and e.is_deleted = false
                """,
                (page_id, int(member_id)),
            )
            c2 = int(cur.fetchone()["c"])
            return c1 + c2
    finally:
        conn.close()


def rename_member(page_id, member_id: int, new_name: str) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    new_name = (new_name or "").strip()
    if not new_name:
        return False, "Name is empty"

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    update members
                    set name = %s
                    where page_id = %s and id = %s and is_deleted = false
                    """,
                    (new_name, page_id, int(member_id)),
                )
                if cur.rowcount == 0:
                    conn.rollback()
                    return False, "Member not found"
                conn.commit()
                return True, "Updated"
            except pg_errors.UniqueViolation:
                conn.rollback()
                return False, "That name already exists"
    finally:
        conn.close()


def soft_delete_member_everywhere(page_id, member_id: int) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Expenses paid by this member
            cur.execute(
                """
                update expenses
                set is_deleted = true, deleted_at = now()
                where page_id = %s and paid_by_member_id = %s and is_deleted = false
                """,
                (page_id, int(member_id)),
            )

            # Shares referencing this member inside this page
            cur.execute(
                """
                update expense_shares
                set is_deleted = true, deleted_at = now()
                where member_id = %s and is_deleted = false
                  and expense_id in (select id from expenses where page_id = %s)
                """,
                (int(member_id), page_id),
            )

            # Expenses that now have zero active shares
            cur.execute(
                """
                update expenses
                set is_deleted = true, deleted_at = now()
                where page_id = %s and is_deleted = false
                  and id in (
                    select e.id
                    from expenses e
                    left join expense_shares s
                      on s.expense_id = e.id and s.is_deleted = false
                    where e.page_id = %s and e.is_deleted = false
                    group by e.id
                    having count(s.member_id) = 0
                  )
                """,
                (page_id, page_id),
            )

            # Member itself
            cur.execute(
                """
                update members
                set is_deleted = true, deleted_at = now()
                where page_id = %s and id = %s and is_deleted = false
                """,
                (page_id, int(member_id)),
            )

            conn.commit()
            return True, "Deleted"
    except Exception as e:
        conn.rollback()
        return False, f"Delete failed: {e}"
    finally:
        conn.close()


def restore_member(page_id, member_id: int) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                update members
                set is_deleted = false, deleted_at = null
                where page_id = %s and id = %s and is_deleted = true
                """,
                (page_id, int(member_id)),
            )
            if cur.rowcount == 0:
                conn.rollback()
                return False, "Member not found"
            conn.commit()
            return True, "Restored"
    except Exception as e:
        conn.rollback()
        return False, f"Restore failed: {e}"
    finally:
        conn.close()


# ------------------------
# Expense ops (page scoped)
# ------------------------
def add_expense(
    page_id,
    expense_date: date,
    description: str,
    amount: float,
    currency: str,
    paid_by_member_id: int,
    target_member_ids: List[int],
) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    description = (description or "").strip()
    if not description:
        return False, "Title is empty"
    if amount is None or amount < 0:
        return False, "Invalid amount"
    if currency not in ("USD", "JPY"):
        return False, "Invalid currency"
    if not target_member_ids:
        return False, "Please select targets"

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select is_deleted from members where page_id = %s and id = %s",
                (page_id, int(paid_by_member_id)),
            )
            payer = cur.fetchone()
            if not payer or bool(payer["is_deleted"]):
                conn.rollback()
                return False, "Payer not found"

            cur.execute(
                """
                select count(*) as c
                from members
                where page_id = %s
                  and id = any(%s)
                  and is_deleted = false
                """,
                (page_id, [int(x) for x in target_member_ids]),
            )
            if int(cur.fetchone()["c"]) != len(target_member_ids):
                conn.rollback()
                return False, "Targets include deleted members"

            cur.execute(
                """
                insert into expenses (page_id, expense_date, description, amount, currency, paid_by_member_id)
                values (%s, %s, %s, %s, %s, %s)
                returning id
                """,
                (
                    page_id,
                    expense_date,
                    description,
                    float(amount),
                    currency,
                    int(paid_by_member_id),
                ),
            )
            expense_id = int(cur.fetchone()["id"])

            cur.executemany(
                """
                insert into expense_shares (expense_id, member_id, is_deleted, deleted_at)
                values (%s, %s, false, null)
                """,
                [(expense_id, int(mid)) for mid in target_member_ids],
            )

            conn.commit()
            return True, "Saved"
    except Exception as e:
        conn.rollback()
        return False, f"Save failed: {e}"
    finally:
        conn.close()


def update_expense(
    page_id,
    expense_id: int,
    expense_date: date,
    description: str,
    amount: float,
    currency: str,
    paid_by_member_id: int,
    target_member_ids: List[int],
) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    description = (description or "").strip()
    if not description:
        return False, "Title is empty"
    if amount is None or amount < 0:
        return False, "Invalid amount"
    if currency not in ("USD", "JPY"):
        return False, "Invalid currency"
    if not target_member_ids:
        return False, "Please select targets"

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select is_deleted from expenses where page_id = %s and id = %s",
                (page_id, int(expense_id)),
            )
            ex = cur.fetchone()
            if not ex or bool(ex["is_deleted"]):
                conn.rollback()
                return False, "Expense not found"

            cur.execute(
                "select is_deleted from members where page_id = %s and id = %s",
                (page_id, int(paid_by_member_id)),
            )
            payer = cur.fetchone()
            if not payer or bool(payer["is_deleted"]):
                conn.rollback()
                return False, "Payer not found"

            cur.execute(
                """
                select count(*) as c
                from members
                where page_id = %s
                  and id = any(%s)
                  and is_deleted = false
                """,
                (page_id, [int(x) for x in target_member_ids]),
            )
            if int(cur.fetchone()["c"]) != len(target_member_ids):
                conn.rollback()
                return False, "Targets include deleted members"

            cur.execute(
                """
                update expenses
                set expense_date = %s,
                    description = %s,
                    amount = %s,
                    currency = %s,
                    paid_by_member_id = %s
                where page_id = %s and id = %s and is_deleted = false
                """,
                (
                    expense_date,
                    description,
                    float(amount),
                    currency,
                    int(paid_by_member_id),
                    page_id,
                    int(expense_id),
                ),
            )

            # Soft delete all current shares
            cur.execute(
                """
                update expense_shares
                set is_deleted = true, deleted_at = now()
                where expense_id = %s and is_deleted = false
                """,
                (int(expense_id),),
            )

            # Re add shares with upsert
            for mid in target_member_ids:
                cur.execute(
                    """
                    insert into expense_shares (expense_id, member_id, is_deleted, deleted_at)
                    values (%s, %s, false, null)
                    on conflict (expense_id, member_id)
                    do update set is_deleted = false, deleted_at = null
                    """,
                    (int(expense_id), int(mid)),
                )

            # If no active share remains, delete expense
            cur.execute(
                "select count(*) as c from expense_shares where expense_id = %s and is_deleted = false",
                (int(expense_id),),
            )
            if int(cur.fetchone()["c"]) == 0:
                cur.execute(
                    """
                    update expenses
                    set is_deleted = true, deleted_at = now()
                    where page_id = %s and id = %s
                    """,
                    (page_id, int(expense_id)),
                )

            conn.commit()
            return True, "Updated"
    except Exception as e:
        conn.rollback()
        return False, f"Update failed: {e}"
    finally:
        conn.close()


def soft_delete_expense(page_id, expense_id: int) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                update expenses
                set is_deleted = true, deleted_at = now()
                where page_id = %s and id = %s and is_deleted = false
                """,
                (page_id, int(expense_id)),
            )
            if cur.rowcount == 0:
                conn.rollback()
                return False, "Expense not found"

            cur.execute(
                """
                update expense_shares
                set is_deleted = true, deleted_at = now()
                where expense_id = %s and is_deleted = false
                """,
                (int(expense_id),),
            )

            conn.commit()
            return True, "Deleted"
    except Exception as e:
        conn.rollback()
        return False, f"Delete failed: {e}"
    finally:
        conn.close()


def restore_expense(page_id, expense_id: int) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select paid_by_member_id
                from expenses
                where page_id = %s and id = %s and is_deleted = true
                """,
                (page_id, int(expense_id)),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return False, "Expense not found"

            payer_id = int(row["paid_by_member_id"])
            cur.execute(
                "select is_deleted from members where page_id = %s and id = %s",
                (page_id, payer_id),
            )
            payer = cur.fetchone()
            if not payer or bool(payer["is_deleted"]):
                conn.rollback()
                return False, "Cannot restore: payer is deleted"

            cur.execute(
                """
                update expenses
                set is_deleted = false, deleted_at = null
                where page_id = %s and id = %s and is_deleted = true
                """,
                (page_id, int(expense_id)),
            )

            cur.execute(
                """
                update expense_shares
                set is_deleted = false, deleted_at = null
                where expense_id = %s
                  and member_id in (
                    select id from members
                    where page_id = %s and is_deleted = false
                  )
                """,
                (int(expense_id), page_id),
            )

            cur.execute(
                "select count(*) as c from expense_shares where expense_id = %s and is_deleted = false",
                (int(expense_id),),
            )
            if int(cur.fetchone()["c"]) == 0:
                cur.execute(
                    """
                    update expenses
                    set is_deleted = true, deleted_at = now()
                    where page_id = %s and id = %s
                    """,
                    (page_id, int(expense_id)),
                )
                conn.commit()
                return False, "Cannot restore: no active targets"

            conn.commit()
            return True, "Restored"
    except Exception as e:
        conn.rollback()
        return False, f"Restore failed: {e}"
    finally:
        conn.close()


def fetch_expenses(page_id, active_only: bool = True) -> List[Dict]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if active_only:
                cur.execute(
                    """
                    select
                        e.id,
                        e.expense_date,
                        e.description,
                        e.amount,
                        e.currency,
                        e.paid_by_member_id,
                        e.created_at,
                        m.name as paid_by
                    from expenses e
                    join members m on m.id = e.paid_by_member_id
                    where e.page_id = %s
                      and e.is_deleted = false
                      and m.is_deleted = false
                    order by e.expense_date desc, e.created_at desc, e.id desc
                    """,
                    (page_id,),
                )
            else:
                cur.execute(
                    """
                    select
                        e.id,
                        e.expense_date,
                        e.description,
                        e.amount,
                        e.currency,
                        e.paid_by_member_id,
                        e.created_at,
                        e.is_deleted,
                        e.deleted_at,
                        m.name as paid_by
                    from expenses e
                    join members m on m.id = e.paid_by_member_id
                    where e.page_id = %s
                    order by e.expense_date desc, e.created_at desc, e.id desc
                    """,
                    (page_id,),
                )

            expenses = cur.fetchall()

            for ex in expenses:
                if active_only:
                    cur.execute(
                        """
                        select m.id, m.name
                        from expense_shares s
                        join members m on m.id = s.member_id
                        where s.expense_id = %s
                          and s.is_deleted = false
                          and m.is_deleted = false
                          and m.page_id = %s
                        order by lower(m.name)
                        """,
                        (int(ex["id"]), page_id),
                    )
                    rows = cur.fetchall()
                    ex["target_ids"] = [int(r["id"]) for r in rows]
                    ex["targets"] = [r["name"] for r in rows]
                else:
                    cur.execute(
                        """
                        select m.id, m.name, s.is_deleted
                        from expense_shares s
                        join members m on m.id = s.member_id
                        where s.expense_id = %s
                          and m.page_id = %s
                        order by lower(m.name)
                        """,
                        (int(ex["id"]), page_id),
                    )
                    rows = cur.fetchall()
                    ex["target_ids"] = [int(r["id"]) for r in rows if not bool(r["is_deleted"])]
                    ex["targets"] = [r["name"] for r in rows if not bool(r["is_deleted"])]

            return expenses
    finally:
        conn.close()


def compute_net_balances(page_id) -> Dict[str, Dict[str, float]]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select id, name from members where page_id = %s and is_deleted = false",
                (page_id,),
            )
            members = cur.fetchall()
            id_to_name = {int(r["id"]): r["name"] for r in members}

            balances: Dict[str, Dict[str, float]] = {"USD": {}, "JPY": {}}
            for name in id_to_name.values():
                balances["USD"][name] = 0.0
                balances["JPY"][name] = 0.0

            cur.execute(
                """
                select id, amount, currency, paid_by_member_id
                from expenses
                where page_id = %s
                  and is_deleted = false
                  and paid_by_member_id in (
                    select id from members where page_id = %s and is_deleted = false
                  )
                """,
                (page_id, page_id),
            )
            expenses = cur.fetchall()

            for ex in expenses:
                ex_id = int(ex["id"])
                amount = float(ex["amount"])
                currency = ex["currency"]
                payer_id = int(ex["paid_by_member_id"])

                cur.execute(
                    """
                    select member_id
                    from expense_shares
                    where expense_id = %s
                      and is_deleted = false
                      and member_id in (
                        select id from members where page_id = %s and is_deleted = false
                      )
                    """,
                    (ex_id, page_id),
                )
                targets = [int(r["member_id"]) for r in cur.fetchall()]
                if not targets:
                    continue

                split = amount / len(targets)

                payer_name = id_to_name.get(payer_id)
                if payer_name is None:
                    continue
                balances[currency][payer_name] += amount

                for t in targets:
                    t_name = id_to_name.get(t)
                    if t_name is None:
                        continue
                    balances[currency][t_name] -= split

            return balances
    finally:
        conn.close()


def build_transaction_matrix(page_id, currency: str) -> pd.DataFrame:
    page_id = _norm_page_id(page_id)
    members = get_members(page_id)
    member_names = [m["name"] for m in members]
    if not member_names:
        return pd.DataFrame()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    e.id,
                    e.expense_date,
                    e.created_at,
                    e.description,
                    e.amount,
                    m.name as payer_name
                from expenses e
                join members m on m.id = e.paid_by_member_id
                where e.page_id = %s
                  and e.currency = %s
                  and e.is_deleted = false
                  and m.is_deleted = false
                order by e.expense_date desc, e.created_at desc, e.id desc
                """,
                (page_id, currency),
            )
            exp_rows = cur.fetchall()

            expense_ids = [int(r["id"]) for r in exp_rows]
            targets_map: Dict[int, List[str]] = {eid: [] for eid in expense_ids}

            if expense_ids:
                cur.execute(
                    """
                    select s.expense_id, m.name as target_name
                    from expense_shares s
                    join members m on m.id = s.member_id
                    where s.expense_id = any(%s)
                      and s.is_deleted = false
                      and m.is_deleted = false
                      and m.page_id = %s
                    """,
                    (expense_ids, page_id),
                )
                for r in cur.fetchall():
                    targets_map[int(r["expense_id"])].append(r["target_name"])

        data_rows: List[Dict] = []
        for r in exp_rows:
            eid = int(r["id"])
            when_str = str(r["expense_date"])
            created_at = str(r["created_at"])
            title = r["description"]
            amount = float(r["amount"])
            payer = r["payer_name"]
            targets = targets_map.get(eid, [])
            if not targets:
                continue

            split = amount / len(targets)

            row = {"When": when_str, "Created at": created_at, "Title": title}
            for name in member_names:
                row[name] = 0.0

            row[payer] += amount
            for t in targets:
                row[t] -= split

            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        if df.empty:
            return df

        cols = ["When", "Created at", "Title"] + member_names
        return df[cols]
    finally:
        conn.close()


# ------------------------
# Bulk and Danger ops (page scoped)
# ------------------------
def soft_delete_all_expenses(page_id) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                update expenses
                set is_deleted = true, deleted_at = now()
                where page_id = %s and is_deleted = false
                """,
                (page_id,),
            )
            cur.execute(
                """
                update expense_shares
                set is_deleted = true, deleted_at = now()
                where expense_id in (select id from expenses where page_id = %s)
                  and is_deleted = false
                """,
                (page_id,),
            )
        conn.commit()
        return True, "Deleted all history (soft delete)"
    except Exception as e:
        conn.rollback()
        return False, f"Bulk delete failed: {e}"
    finally:
        conn.close()


def soft_delete_all_members_everywhere(page_id) -> Tuple[bool, str]:
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                update expenses
                set is_deleted = true, deleted_at = now()
                where page_id = %s and is_deleted = false
                """,
                (page_id,),
            )
            cur.execute(
                """
                update expense_shares
                set is_deleted = true, deleted_at = now()
                where expense_id in (select id from expenses where page_id = %s)
                  and is_deleted = false
                """,
                (page_id,),
            )
            cur.execute(
                """
                update members
                set is_deleted = true, deleted_at = now()
                where page_id = %s and is_deleted = false
                """,
                (page_id,),
            )
        conn.commit()
        return True, "Deleted all members and history (soft delete)"
    except Exception as e:
        conn.rollback()
        return False, f"Bulk delete failed: {e}"
    finally:
        conn.close()


def wipe_page(page_id) -> Tuple[bool, str]:
    """
    Delete ALL data for a single page (members, expenses, shares, and the page itself).
    This does NOT delete the whole database.
    """
    page_id = _norm_page_id(page_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("select id from pages where id = %s and is_deleted = false", (page_id,))
            if not cur.fetchone():
                conn.rollback()
                return False, "Page not found"

            # shares -> expenses -> members -> page
            cur.execute(
                """
                delete from expense_shares
                where expense_id in (select id from expenses where page_id = %s)
                """,
                (page_id,),
            )
            cur.execute("delete from expenses where page_id = %s", (page_id,))
            cur.execute("delete from members where page_id = %s", (page_id,))
            cur.execute("delete from pages where id = %s", (page_id,))

        conn.commit()
        return True, "Deleted this page and all its data"
    except Exception as e:
        conn.rollback()
        return False, f"Wipe failed: {e}"
    finally:
        conn.close()


def delete_db_file() -> Tuple[bool, str]:
    # No local db file in Supabase mode
    return False, "Not supported in Supabase mode"