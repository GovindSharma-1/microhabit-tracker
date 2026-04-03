"""
MicroHabit — Single-file Streamlit micro habit tracker.
Placement-ready: OOP data layer, SQLite persistence, error handling, modern UI.
"""

from __future__ import annotations

import html
import random
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parent / "habits.db"

MOTIVATIONAL_QUOTES: list[str] = [
    "Small steps every day add up to big change. 🌱",
    "Progress, not perfection — you've got this. ✨",
    "Consistency beats intensity. Show up again today. 💪",
    "Your future self will thank you for today's effort. 🙏",
    "One micro habit at a time builds an unstoppable you. 🚀",
    "Discipline is choosing what you want most over what you want now. 🎯",
    "Every checkmark is a vote for the person you're becoming. ✅",
    "Start where you are. Use what you have. Do what you can. 🌿",
    "The best time to plant a tree was years ago. The second best is today. 🌳",
    "Tiny improvements compound into remarkable results. 📈",
]


def daily_quote(today: date) -> str:
    """Deterministic quote per calendar day (feels fresh but stable within the day)."""
    rng = random.Random(today.toordinal())
    return rng.choice(MOTIVATIONAL_QUOTES)


# ---------------------------------------------------------------------------
# Data models (lightweight rows for UI)
# ---------------------------------------------------------------------------


@dataclass
class HabitRow:
    id: int
    name: str
    description: Optional[str]
    created_at: str


@dataclass
class TodayHabitView:
    habit: HabitRow
    done_today: bool


@dataclass
class HabitPrediction:
    """Output of predict_completion_probability for the Smart Prediction tab."""

    habit_id: int
    habit_name: str
    today_pct: float
    risk_level: str  # "Low" | "Medium" | "High"
    risk_emoji: str
    insight: str
    tip: str


# ---------------------------------------------------------------------------
# HabitTracker — SQLite-backed domain layer
# ---------------------------------------------------------------------------


class HabitTracker:
    """
    Persists habits and daily completion logs in SQLite.
    Tables: habits (id, name, description, created_at), daily_logs (habit_id, log_date).
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._ensure_parent()
        self._init_db()

    def _ensure_parent(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS habits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        habit_id INTEGER NOT NULL,
                        log_date TEXT NOT NULL,
                        UNIQUE (habit_id, log_date),
                        FOREIGN KEY (habit_id) REFERENCES habits(id) ON DELETE CASCADE
                    )
                    """
                )
        except sqlite3.Error as e:
            raise RuntimeError(f"Database initialization failed: {e}") from e

    def add_habit(self, name: str, description: Optional[str] = None) -> int:
        """Insert a habit; returns new id. Raises on duplicate name or DB error."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Habit name cannot be empty.")
        desc = (description or "").strip() or None
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO habits (name, description) VALUES (?, ?)",
                    (name, desc),
                )
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            raise ValueError("A habit with this name already exists.") from e
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not add habit: {e}") from e

    def mark_done(self, habit_id: int, log_date: Optional[date] = None) -> None:
        """Record completion for habit on log_date (default: today)."""
        d = (log_date or date.today()).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO daily_logs (habit_id, log_date) VALUES (?, ?)",
                    (habit_id, d),
                )
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not mark habit done: {e}") from e

    def mark_undone(self, habit_id: int, log_date: Optional[date] = None) -> None:
        """Remove completion for habit on log_date (default: today)."""
        d = (log_date or date.today()).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM daily_logs WHERE habit_id = ? AND log_date = ?",
                    (habit_id, d),
                )
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not update habit: {e}") from e

    def set_done_today(self, habit_id: int, done: bool) -> None:
        """Idempotent UI helper: sync checkbox to DB for today."""
        if done:
            self.mark_done(habit_id)
        else:
            self.mark_undone(habit_id)

    def get_all_habits(self) -> list[HabitRow]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, name, description, created_at FROM habits ORDER BY name COLLATE NOCASE"
                ).fetchall()
            return [
                HabitRow(
                    id=r["id"],
                    name=r["name"],
                    description=r["description"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not load habits: {e}") from e

    def is_done_on(self, habit_id: int, d: date) -> bool:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM daily_logs WHERE habit_id = ? AND log_date = ? LIMIT 1",
                    (habit_id, d.isoformat()),
                ).fetchone()
            return row is not None
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not read log: {e}") from e

    def get_today_progress(self, today: Optional[date] = None) -> list[TodayHabitView]:
        """All habits with whether each is completed on `today` (default: system today)."""
        day = today or date.today()
        habits = self.get_all_habits()
        return [
            TodayHabitView(habit=h, done_today=self.is_done_on(h.id, day)) for h in habits
        ]

    def get_weekly_progress(self, end: Optional[date] = None) -> list[dict[str, Any]]:
        """
        Last 7 days ending at `end` (inclusive): per-day completion rate.
        Each item: {"date": date, "pct": float 0-100, "done": int, "total": int}.
        """
        last = end or date.today()
        habits = self.get_all_habits()
        n = len(habits)
        if n == 0:
            return [
                {"date": last - timedelta(days=i), "pct": 0.0, "done": 0, "total": 0}
                for i in range(6, -1, -1)
            ]

        try:
            with self._connect() as conn:
                out: list[dict[str, Any]] = []
                for i in range(6, -1, -1):
                    d = last - timedelta(days=i)
                    ds = d.isoformat()
                    # Habits that existed on d: created_at date <= d
                    cur = conn.execute(
                        """
                        SELECT COUNT(*) FROM habits
                        WHERE date(created_at) <= date(?)
                        """,
                        (ds,),
                    )
                    total = int(cur.fetchone()[0])
                    if total == 0:
                        out.append({"date": d, "pct": 0.0, "done": 0, "total": 0})
                        continue
                    cur = conn.execute(
                        """
                        SELECT COUNT(DISTINCT habit_id) FROM daily_logs
                        WHERE log_date = ?
                          AND habit_id IN (
                            SELECT id FROM habits WHERE date(created_at) <= date(?)
                          )
                        """,
                        (ds, ds),
                    )
                    done = int(cur.fetchone()[0])
                    pct = (done / total) * 100.0 if total else 0.0
                    out.append({"date": d, "pct": round(pct, 1), "done": done, "total": total})
                return out
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not load weekly progress: {e}") from e

    def get_current_streak(self, today: Optional[date] = None) -> int:
        """
        Consecutive calendar days ending at `today` where every habit that existed
        that day was completed (100% perfect days). Streak breaks on first imperfect day.
        """
        day = today or date.today()
        habits = self.get_all_habits()
        if not habits:
            return 0

        streak = 0
        d = day
        try:
            with self._connect() as conn:
                while True:
                    ds = d.isoformat()
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM habits WHERE date(created_at) <= date(?)",
                        (ds,),
                    )
                    total = int(cur.fetchone()[0])
                    if total == 0:
                        break
                    cur = conn.execute(
                        """
                        SELECT COUNT(DISTINCT habit_id) FROM daily_logs
                        WHERE log_date = ?
                          AND habit_id IN (
                            SELECT id FROM habits WHERE date(created_at) <= date(?)
                          )
                        """,
                        (ds, ds),
                    )
                    done = int(cur.fetchone()[0])
                    if done < total:
                        break
                    streak += 1
                    d -= timedelta(days=1)
                    # Optional safety: avoid infinite loop if clock weird
                    if streak > 10000:
                        break
            return streak
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not compute streak: {e}") from e

    def get_overall_completion_pct(self, today: Optional[date] = None) -> float:
        """
        Average daily completion % from first habit creation through `today`
        (only days on which at least one habit existed).
        """
        day = today or date.today()
        habits = self.get_all_habits()
        if not habits:
            return 0.0

        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT date(MIN(created_at)) FROM habits"
                ).fetchone()
                if not row or row[0] is None:
                    return 0.0
                start = datetime.strptime(row[0], "%Y-%m-%d").date()
                if start > day:
                    return 0.0

                total_pct = 0.0
                count_days = 0
                d = start
                while d <= day:
                    ds = d.isoformat()
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM habits WHERE date(created_at) <= date(?)",
                        (ds,),
                    )
                    total = int(cur.fetchone()[0])
                    if total > 0:
                        cur = conn.execute(
                            """
                            SELECT COUNT(DISTINCT habit_id) FROM daily_logs
                            WHERE log_date = ?
                              AND habit_id IN (
                                SELECT id FROM habits WHERE date(created_at) <= date(?)
                              )
                            """,
                            (ds, ds),
                        )
                        done = int(cur.fetchone()[0])
                        total_pct += (done / total) * 100.0
                        count_days += 1
                    d += timedelta(days=1)

            return round(total_pct / count_days, 1) if count_days else 0.0
        except (sqlite3.Error, ValueError) as e:
            raise RuntimeError(f"Could not compute overall completion: {e}") from e

    # --- Smart prediction (per-habit) ---------------------------------------

    def _habit_created_date(self, habit_id: int) -> date:
        """First calendar day the habit exists (from created_at)."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT date(created_at) FROM habits WHERE id = ?",
                    (habit_id,),
                ).fetchone()
            if not row or row[0] is None:
                raise ValueError(f"No habit with id {habit_id}.")
            return datetime.strptime(row[0], "%Y-%m-%d").date()
        except sqlite3.Error as e:
            raise RuntimeError(f"Could not read habit: {e}") from e

    def _trailing_completion_streak(self, habit_id: int, ref: date) -> int:
        """
        Consecutive days with a log, anchored at the most recent completion on or before ref.
        If ref is not logged, streak counts backward from the last logged day before ref.
        """
        created = self._habit_created_date(habit_id)
        if ref < created:
            return 0
        cur = ref
        while cur >= created and not self.is_done_on(habit_id, cur):
            cur -= timedelta(days=1)
        if cur < created:
            return 0
        streak = 0
        while cur >= created and self.is_done_on(habit_id, cur):
            streak += 1
            cur -= timedelta(days=1)
        return streak

    def _consistency_last_7(self, habit_id: int, end: date) -> float:
        """Fraction of the 7 calendar days ending at `end` (inclusive) that have a log (0.0–1.0)."""
        created = self._habit_created_date(habit_id)
        start_win = end - timedelta(days=6)
        done_days = 0
        total = 0
        d = start_win
        while d <= end:
            if d >= created:
                total += 1
                if self.is_done_on(habit_id, d):
                    done_days += 1
            d += timedelta(days=1)
        return (done_days / total) if total else 0.0

    def _overall_habit_rate(self, habit_id: int, end: date) -> float:
        """
        Historical completion rate for this habit: logged days / eligible days from creation through end.
        """
        created = self._habit_created_date(habit_id)
        if end < created:
            return 0.0
        eligible = 0
        logged = 0
        d = created
        while d <= end:
            eligible += 1
            if self.is_done_on(habit_id, d):
                logged += 1
            d += timedelta(days=1)
        return (logged / eligible) if eligible else 0.0

    def _base_probability_pct(
        self,
        streak_days: int,
        consistency_7: float,
        overall_rate: float,
    ) -> float:
        """
        Weighted blend (all inputs on 0–1 scale except streak, which is normalized).

        Formula (requirement):
            probability = (0.4 * streak_score)
                        + (0.4 * consistency_7day)
                        + (0.2 * overall_rate)

        streak_score maps consecutive completion days to [0, 1]; 14+ days => 1.0.
        Result is multiplied by 100 for a percentage before noise is applied.
        """
        streak_score = min(streak_days / 14.0, 1.0)
        blended = (
            0.4 * streak_score
            + 0.4 * consistency_7
            + 0.2 * overall_rate
        )
        return max(0.0, min(100.0, blended * 100.0))

    @staticmethod
    def _apply_prediction_noise(base_pct: float) -> float:
        """±5% jitter so predictions feel human / non-deterministic; clamp to [0, 100]."""
        jitter = random.uniform(-5.0, 5.0)
        return max(0.0, min(100.0, base_pct + jitter))

    def _missed_days_last_7(self, habit_id: int, end: date) -> int:
        """Count eligible days in the 7-day window ending `end` with no completion log."""
        created = self._habit_created_date(habit_id)
        start = end - timedelta(days=6)
        missed = 0
        d = start
        while d <= end:
            if d >= created and not self.is_done_on(habit_id, d):
                missed += 1
            d += timedelta(days=1)
        return missed

    @staticmethod
    def _classify_risk(
        done_today: bool,
        today_pct_noisy: float,
        consistency_7: float,
        streak_days: int,
        missed_last_7: int,
    ) -> tuple[str, str]:
        """
        Habit Insights & Risk Alert — discrete risk level from signals + noisy probability.

        Uses:
        - **done_today**: if True, slip risk for *today* is resolved → Low.
        - **today_pct_noisy**: post-noise completion probability (primary gauge).
        - **consistency_7**: share of last 7 days completed (0–1); low values raise risk.
        - **streak_days**: longer trailing streak lowers risk when paired with decent consistency.
        - **missed_last_7**: raw count of non-completed eligible days in the window; more misses → higher risk.

        Rules (tuned for clear UX):
        - **High**: weak probability, very weak week, or many misses.
        - **Low**: strong probability + tolerable misses, or already done today.
        - **Medium**: everything in between.
        """
        if done_today:
            return "Low", "🟢"
        if today_pct_noisy < 48.0 or consistency_7 <= (2.0 / 7.0) or missed_last_7 >= 4:
            return "High", "🔴"
        if (
            today_pct_noisy >= 72.0
            and consistency_7 >= (4.0 / 7.0)
            and missed_last_7 <= 1
        ):
            return "Low", "🟢"
        if (
            today_pct_noisy >= 60.0
            and consistency_7 >= (5.0 / 7.0)
            and missed_last_7 <= 2
            and streak_days >= 3
        ):
            return "Low", "🟢"
        if today_pct_noisy < 58.0 or missed_last_7 >= 3:
            return "High", "🔴"
        return "Medium", "🟠"

    @staticmethod
    def _personalized_insight(
        habit_name: str,
        done_today: bool,
        streak_days: int,
        consistency_7: float,
        missed_last_7: int,
    ) -> str:
        """One line tailored to streak + 7-day pattern (not the random % jitter)."""
        if done_today:
            return f"Nice — «{habit_name}» is done for today. Carry this win forward. ✨"
        if streak_days >= 7 and consistency_7 >= (6.0 / 7.0):
            return "Strong 7-day streak – you're building momentum! 🔥"
        if missed_last_7 >= 4:
            return "Pattern shows several missed days — shrink the habit to 2 minutes and log once. 🎯"
        if missed_last_7 in (2, 3):
            return f"You've missed {missed_last_7} days recently – try setting a reminder. ⏰"
        if streak_days >= 3 and consistency_7 >= (4.0 / 7.0):
            return "You're showing up more often than not — one more check-in locks the habit. ✅"
        if streak_days == 0 and consistency_7 < (3.0 / 7.0):
            return "Fresh runway — one completion today breaks the cold streak. 🌱"
        return "Small steps beat zero — pick the easiest version of this habit now. 💪"

    @staticmethod
    def _motivational_tip(
        habit_name: str,
        risk_level: str,
        habit_id: int,
        today: date,
    ) -> str:
        """
        One actionable tip, stable for the same habit on the same day (deterministic RNG)
        so the tab doesn't flicker on every Streamlit rerun.
        """
        rng = random.Random(habit_id * 100_003 + today.toordinal())
        low_tips = [
            "Keep the chain visible — mark {name} right after a fixed daily anchor (breakfast, login, bedtime).",
            "Protect what works: same time, same cue for {name} this week.",
            "You're in a groove — write one sentence in a note when you finish {name} to reinforce the win.",
        ]
        mid_tips = [
            "Make {name} stupid-easy: lower the bar to 60 seconds so \"later\" never wins.",
            "Stack {name} onto something you never skip (shower, first coffee, shoes on).",
            "If you miss once, show up the next day — never twice in a row for {name}.",
        ]
        high_tips = [
            "Set one phone alarm labeled «{name}» — friction beats forgetting.",
            "Pre-decide the minimum: e.g. one sip, one page, one minute for {name}.",
            "Tell someone you'll report {name} tonight — accountability nudges follow-through.",
            "Remove one distraction before {name} (silent mode, tab closed) — environment shapes behavior.",
        ]
        pool = {"Low": low_tips, "Medium": mid_tips, "High": high_tips}[risk_level]
        template = rng.choice(pool)
        return template.format(name=habit_name)

    def predict_completion_probability(self, habit_id: int) -> HabitPrediction:
        """
        Today's completion probability plus Habit Insights & Risk Alert (no tomorrow forecast).

        **Probability (before ±5% noise)** uses the same blend as before:
            0.4 * streak_score + 0.4 * consistency_7day + 0.2 * overall_rate
        If the habit is already logged today, base probability is 100% (then noise).

        **Risk level** combines the *noisy* probability with 7-day consistency, streak length,
        and missed-day count — see `_classify_risk` docstring.

        **Insight** summarizes streak + weekly pattern; **tip** is a single actionable suggestion
        matched to risk level (deterministic per habit per day).
        """
        today = date.today()
        habits = {h.id: h for h in self.get_all_habits()}
        if habit_id not in habits:
            raise ValueError(f"No habit with id {habit_id}.")

        hrow = habits[habit_id]
        name = hrow.name
        done_today = self.is_done_on(habit_id, today)

        streak_today = self._trailing_completion_streak(habit_id, today)
        c7_today = self._consistency_last_7(habit_id, today)
        overall_today = self._overall_habit_rate(habit_id, today)
        missed_7 = self._missed_days_last_7(habit_id, today)

        if done_today:
            base_today = 100.0
        else:
            base_today = self._base_probability_pct(streak_today, c7_today, overall_today)

        today_pct = round(self._apply_prediction_noise(base_today), 1)

        risk_level, risk_emoji = self._classify_risk(
            done_today, today_pct, c7_today, streak_today, missed_7
        )
        insight = self._personalized_insight(
            name, done_today, streak_today, c7_today, missed_7
        )
        tip = self._motivational_tip(name, risk_level, habit_id, today)

        return HabitPrediction(
            habit_id=habit_id,
            habit_name=name,
            today_pct=today_pct,
            risk_level=risk_level,
            risk_emoji=risk_emoji,
            insight=insight,
            tip=tip,
        )


# ---------------------------------------------------------------------------
# UI: theme & layout
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    """Dark theme with green accents (Streamlit overrides)."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

            html, body, [class*="css"] {
                font-family: 'DM Sans', system-ui, sans-serif;
            }

            .stApp {
                background: linear-gradient(165deg, #0d1117 0%, #0f172a 45%, #0c1419 100%);
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
                border-right: 1px solid rgba(34, 197, 94, 0.15);
            }

            section[data-testid="stSidebar"] .stMarkdown h1,
            section[data-testid="stSidebar"] .stMarkdown h2,
            section[data-testid="stSidebar"] .stMarkdown h3 {
                color: #ecfdf5 !important;
            }

            div[data-testid="stHeader"] {
                background-color: transparent;
            }

            .microhabit-hero {
                background: linear-gradient(135deg, rgba(22, 163, 74, 0.12) 0%, rgba(16, 185, 129, 0.08) 100%);
                border: 1px solid rgba(34, 197, 94, 0.25);
                border-radius: 16px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1.25rem;
            }
            .microhabit-hero h1 {
                color: #ecfdf5;
                font-size: 1.75rem;
                font-weight: 700;
                margin: 0 0 0.35rem 0;
                letter-spacing: -0.02em;
            }
            .microhabit-hero p {
                color: #a7f3d0;
                margin: 0;
                font-size: 1rem;
                line-height: 1.5;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background-color: rgba(15, 23, 42, 0.6);
                border-radius: 12px;
                padding: 6px;
                border: 1px solid rgba(34, 197, 94, 0.12);
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 10px;
                color: #94a3b8;
                font-weight: 600;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #15803d 0%, #059669 100%) !important;
                color: #ecfdf5 !important;
            }

            div[data-testid="stMetric"] {
                background: rgba(30, 41, 59, 0.5);
                border: 1px solid rgba(34, 197, 94, 0.15);
                border-radius: 12px;
                padding: 0.75rem 1rem;
            }
            div[data-testid="stMetric"] label {
                color: #86efac !important;
            }
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: #ecfdf5 !important;
            }

            .stButton > button {
                background: linear-gradient(135deg, #16a34a 0%, #059669 100%);
                color: white;
                border: none;
                font-weight: 600;
                border-radius: 10px;
                padding: 0.5rem 1.25rem;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .stButton > button:hover {
                box-shadow: 0 4px 20px rgba(34, 197, 94, 0.35);
                transform: translateY(-1px);
                color: white;
            }

            .stTextInput input, .stTextArea textarea {
                background-color: rgba(30, 41, 59, 0.8) !important;
                color: #f1f5f9 !important;
                border: 1px solid rgba(34, 197, 94, 0.2) !important;
                border-radius: 10px !important;
            }

            div[data-testid="stCheckbox"] label {
                color: #e2e8f0 !important;
            }

            /* Dataframe / table readability */
            [data-testid="stDataFrame"] {
                border: 1px solid rgba(34, 197, 94, 0.15);
                border-radius: 12px;
                overflow: hidden;
            }

            /* Smart Prediction tab */
            .smart-pred-card {
                background: rgba(30, 41, 59, 0.45);
                border: 1px solid rgba(34, 197, 94, 0.14);
                border-radius: 14px;
                padding: 1.1rem 1.25rem;
                margin-bottom: 1rem;
            }
            .smart-pred-card h4 {
                color: #ecfdf5;
                margin: 0 0 0.5rem 0;
                font-size: 1.1rem;
            }
            .smart-pred-card .pred-meta {
                color: #94a3b8;
                font-size: 0.9rem;
                margin: 0.35rem 0 0.75rem 0;
                line-height: 1.45;
            }
            .pred-row-label {
                color: #86efac;
                font-size: 0.8rem;
                font-weight: 600;
                margin-top: 0.5rem;
                margin-bottom: 0.25rem;
            }
            .pred-bar-track {
                width: 100%;
                height: 10px;
                border-radius: 999px;
                background: rgba(15, 23, 42, 0.9);
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.12);
            }
            .pred-bar-fill {
                height: 100%;
                border-radius: 999px;
                transition: width 0.35s ease;
            }
            .pred-bar-fill.pred-high {
                background: linear-gradient(90deg, #15803d, #4ade80);
            }
            .pred-bar-fill.pred-mid {
                background: linear-gradient(90deg, #a16207, #eab308);
            }
            .pred-bar-fill.pred-low {
                background: linear-gradient(90deg, #991b1b, #f87171);
            }
            .risk-pill-low { color: #86efac; font-weight: 700; }
            .risk-pill-mid { color: #fb923c; font-weight: 700; }
            .risk-pill-high { color: #f87171; font-weight: 700; }
            .smart-pred-card .insight-line {
                color: #cbd5e1;
                font-size: 0.95rem;
                margin: 0.5rem 0 0.35rem 0;
                line-height: 1.5;
            }
            .smart-pred-card .tip-line {
                color: #94a3b8;
                font-size: 0.88rem;
                margin: 0.35rem 0 0 0;
                line-height: 1.45;
                border-left: 3px solid rgba(34, 197, 94, 0.35);
                padding-left: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_weekly_chart(weekly: list[dict[str, Any]]) -> go.Figure:
    labels = [w["date"].strftime("%a %d") for w in weekly]
    values = [w["pct"] for w in weekly]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(
                    color=values,
                    colorscale=[[0, "#14532d"], [0.5, "#16a34a"], [1, "#4ade80"]],
                    line=dict(width=0),
                ),
                text=[f"{v}%" for v in values],
                textposition="outside",
                textfont=dict(color="#a7f3d0", size=12),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#cbd5e1", family="DM Sans, sans-serif"),
        margin=dict(t=40, b=48, l=40, r=24),
        yaxis=dict(
            title="Completion %",
            range=[0, 105],
            gridcolor="rgba(34,197,94,0.12)",
            zeroline=False,
        ),
        xaxis=dict(gridcolor="rgba(148,163,184,0.08)"),
        showlegend=False,
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def get_tracker() -> HabitTracker:
    """
    Return a fresh HabitTracker each script run.

    We intentionally do **not** cache this with @st.cache_resource: Streamlit’s resource
    cache can keep an old class/instance across edits, so the UI looks like “nothing changed.”
    Instantiating HabitTracker is cheap (DB opens per query anyway).
    """
    return HabitTracker(DB_PATH)


def main() -> None:
    st.set_page_config(
        page_title="MicroHabit",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    try:
        tracker = get_tracker()
    except Exception as e:
        st.error(f"Could not open database: {e}")
        st.stop()

    today = date.today()
    quote = daily_quote(today)

    # --- Sidebar: add habit ---
    with st.sidebar:
        st.markdown("### 🌱 MicroHabit")
        st.caption("Build tiny habits that stick.")
        st.divider()
        st.markdown("#### ➕ New habit")
        with st.form("add_habit_form", clear_on_submit=True):
            name_in = st.text_input("Name", placeholder="e.g. Drink water", max_chars=120)
            desc_in = st.text_area(
                "Description (optional)",
                placeholder="Why it matters…",
                max_chars=500,
                height=100,
            )
            submitted = st.form_submit_button("Add habit")
            if submitted:
                try:
                    tracker.add_habit(name_in, desc_in)
                    st.success("Habit added! ✅")
                    st.rerun()
                except ValueError as ve:
                    st.warning(str(ve))
                except RuntimeError as re:
                    st.error(str(re))

        st.divider()
        st.caption(f"DB: `{DB_PATH.name}` · SQLite")

    # --- Main hero ---
    st.markdown(
        f"""
        <div class="microhabit-hero">
            <h1>🌱 MicroHabit</h1>
            <p>{quote}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_today, tab_progress, tab_all, tab_smart = st.tabs(
        ["📋 Today's Habits", "📊 Progress", "📑 All Habits", "🧠 Smart Prediction"]
    )

    # --- Tab: Today ---
    with tab_today:
        try:
            views = tracker.get_today_progress(today)
        except RuntimeError as e:
            st.error(str(e))
            views = []

        if not views:
            st.info("No habits yet. Add one from the sidebar to get started! 🌱")
        else:
            st.markdown(f"**{today.strftime('%A, %d %B %Y')}** — check off what you did.")
            for v in views:
                key = f"habit_done_{v.habit.id}_{today.isoformat()}"
                prev = v.done_today
                checked = st.checkbox(
                    f"**{v.habit.name}**" + (" ✨" if v.done_today else ""),
                    value=prev,
                    key=key,
                    help=v.habit.description or "No description",
                )
                if checked != prev:
                    try:
                        tracker.set_done_today(v.habit.id, checked)
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))

    # --- Tab: Progress ---
    with tab_progress:
        col_a, col_b, col_c = st.columns(3)
        try:
            streak = tracker.get_current_streak(today)
            weekly = tracker.get_weekly_progress(today)
            overall = tracker.get_overall_completion_pct(today)
            n_habits = len(tracker.get_all_habits())
        except RuntimeError as e:
            st.error(str(e))
            streak, weekly, overall, n_habits = 0, [], 0.0, 0

        with col_a:
            st.metric("🔥 Current streak", f"{streak} day{'s' if streak != 1 else ''}")
        with col_b:
            st.metric("✅ Overall completion", f"{overall}%")
        with col_c:
            st.metric("🌿 Active habits", n_habits)

        st.markdown("##### Weekly completion")
        if not weekly or all(w["total"] == 0 for w in weekly):
            st.caption("Add habits and log a few days to see your chart.")
        fig = render_weekly_chart(weekly)
        st.plotly_chart(fig, width="stretch")

    # --- Tab: All habits ---
    with tab_all:
        try:
            all_h = tracker.get_all_habits()
        except RuntimeError as e:
            st.error(str(e))
            all_h = []

        if not all_h:
            st.info("No habits in the database yet.")
        else:
            df = pd.DataFrame(
                [
                    {
                        "Name": h.name,
                        "Description": h.description or "—",
                        "Created": h.created_at[:10] if h.created_at else "—",
                    }
                    for h in all_h
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)

    # --- Tab: Smart Prediction (Habit Insights & Risk Alert) ----------------
    with tab_smart:
        st.markdown("##### 🧠 Smart Prediction · Habit Insights & Risk Alert")
        st.caption(
            "Today's likelihood blends **streak**, **last 7 days**, and **history**, "
            "with **±5%** jitter. **Risk** layers consistency and missed days on top — not just the %."
        )
        try:
            pred_habits = tracker.get_all_habits()
        except RuntimeError as e:
            st.error(str(e))
            pred_habits = []

        if not pred_habits:
            st.info("Add habits to see insights and risk alerts. 🌱")
        else:
            for h in pred_habits:
                try:
                    pr = tracker.predict_completion_probability(h.id)
                except (ValueError, RuntimeError) as e:
                    st.warning(f"{h.name}: {e}")
                    continue

                bar_class = (
                    "pred-high"
                    if pr.today_pct >= 70
                    else "pred-mid"
                    if pr.today_pct >= 55
                    else "pred-low"
                )
                risk_class = (
                    "risk-pill-low"
                    if pr.risk_level == "Low"
                    else "risk-pill-mid"
                    if pr.risk_level == "Medium"
                    else "risk-pill-high"
                )
                safe_name = html.escape(pr.habit_name)
                safe_insight = html.escape(pr.insight)
                safe_tip = html.escape(pr.tip)

                st.markdown(
                    f"""
                    <div class="smart-pred-card">
                        <h4>{safe_name}</h4>
                        <p class="pred-meta">
                            <span class="{risk_class}">Risk: {pr.risk_level} {pr.risk_emoji}</span>
                            · Today's completion · {pr.today_pct:.0f}%
                        </p>
                        <div class="pred-bar-track">
                            <div class="pred-bar-fill {bar_class}" style="width: {pr.today_pct}%;"></div>
                        </div>
                        <p class="insight-line"><strong>Insight:</strong> {safe_insight}</p>
                        <p class="tip-line"><strong>Tip:</strong> {safe_tip}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
