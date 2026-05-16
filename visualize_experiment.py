import ast
import json
import pandas as pd
import matplotlib.pyplot as plt


STEPS_PATH = "steps.csv"
EVENTS_PATH = "events.csv"

def to_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return bool(value)

def parse_event_details(value):
    """
    Parses the details column from events.csv.
    It may look like:
    {"frame": 113, "door": "living_room", "is_open": true}
    """
    if pd.isna(value):
        return {}

    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return {}


def parse_door_states(value):
    """
    Parses the door_states column from steps.csv.
    It may look like:
    {'living_room': False, 'bathroom': False}
    """
    if pd.isna(value):
        return {}

    try:
        return ast.literal_eval(value)
    except Exception:
        return {}


def load_data():
    steps = pd.read_csv("experiment_logs/dynamic_ekf_dynamic_20260515_173157/steps.csv")
    events = pd.read_csv("experiment_logs/dynamic_ekf_dynamic_20260515_173157/events.csv")

    # Parse event details into separate columns
    details = events["details"].apply(parse_event_details)
    events["door"] = details.apply(lambda d: d.get("door"))
    events["is_open"] = details.apply(lambda d: d.get("is_open"))
    events["event_frame"] = details.apply(lambda d: d.get("frame"))

    # Use a clean numeric time column
    steps["timestamp"] = pd.to_numeric(steps["timestamp"], errors="coerce")
    events["time"] = pd.to_numeric(events["time"], errors="coerce")

    return steps, events


def downsample(df, max_points=5000):
    """
    Makes plotting huge files faster.
    Keeps roughly max_points rows.
    """
    if len(df) <= max_points:
        return df

    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()

def get_any_door_open_intervals(events, end_time=None):
    door_events = events[events["event_type"] == "door_state_change"].copy()
    door_events = door_events.sort_values("time")

    door_states = {}
    intervals = []

    currently_any_open = False
    open_start = None

    for _, row in door_events.iterrows():
        door = row["door"]
        is_open = to_bool(row["is_open"])
        t = row["time"]

        door_states[door] = is_open
        any_open = any(door_states.values())

        if any_open and not currently_any_open:
            open_start = t

        if currently_any_open and not any_open:
            intervals.append((open_start, t))
            open_start = None

        currently_any_open = any_open

    if currently_any_open and open_start is not None:
        if end_time is None:
            end_time = door_events["time"].max()
        intervals.append((open_start, end_time))

    return intervals

def plot_coverage_with_door_periods(steps, events):
    plot_steps = downsample(steps)

    plt.figure(figsize=(14, 6))

    plt.plot(
        plot_steps["timestamp"],
        plot_steps["coverage"],
        label="Coverage",
        linewidth=2
    )

    open_intervals = get_any_door_open_intervals(
        events,
        end_time=steps["timestamp"].max()
    )

    for i, (start, end) in enumerate(open_intervals):
        plt.axvspan(
            start,
            end,
            alpha=0.15,
            label="Door-open period" if i == 0 else None
        )

    plt.title("Coverage over Time with Door-Open Periods")
    plt.xlabel("Time")
    plt.ylabel("Coverage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_error_with_door_periods(steps, events):
    plot_steps = downsample(steps)

    plt.figure(figsize=(14, 6))

    plt.plot(
        plot_steps["timestamp"],
        plot_steps["error"],
        label="Current localization error",
        linewidth=1.5
    )

    plt.plot(
        plot_steps["timestamp"],
        plot_steps["avg_error"],
        label="Average localization error",
        linewidth=2
    )

    open_intervals = get_any_door_open_intervals(
        events,
        end_time=steps["timestamp"].max()
    )

    for i, (start, end) in enumerate(open_intervals):
        plt.axvspan(
            start,
            end,
            alpha=0.15,
            label="Door-open period" if i == 0 else None
        )

    plt.title("Localization Error over Time with Door-Open Periods")
    plt.xlabel("Time")
   


def plot_visible_landmarks(steps, events):
    plot_steps = downsample(steps)

    plt.figure(figsize=(14, 6))
    plt.plot(plot_steps["timestamp"], plot_steps["visible_landmarks"], label="Visible landmarks")

    door_events = events[events["event_type"] == "door_state_change"]

    for _, row in door_events.iterrows():
        color = "green" if row["is_open"] else "red"
        linestyle = "--" if row["is_open"] else ":"
        plt.axvline(row["time"], color=color, linestyle=linestyle, alpha=0.25)

    plt.title("Visible Landmarks over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of visible landmarks")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_robot_trajectory(steps):
    plot_steps = downsample(steps)

    plt.figure(figsize=(8, 8))
    plt.plot(plot_steps["robot_x"], plot_steps["robot_y"], label="True robot path")
    plt.plot(plot_steps["localized_x"], plot_steps["localized_y"], label="EKF estimated path", alpha=0.8)

    plt.scatter(
        plot_steps["robot_x"].iloc[0],
        plot_steps["robot_y"].iloc[0],
        marker="o",
        label="Start"
    )

    plt.scatter(
        plot_steps["robot_x"].iloc[-1],
        plot_steps["robot_y"].iloc[-1],
        marker="x",
        label="End"
    )

    plt.title("Robot Trajectory: True Pose vs Localized Pose")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_door_timeline(events):
    door_events = events[events["event_type"] == "door_state_change"].copy()

    if door_events.empty:
        print("No door_state_change events found.")
        return

    doors = sorted(door_events["door"].dropna().unique())

    plt.figure(figsize=(14, 6))

    for i, door in enumerate(doors):
        door_data = door_events[door_events["door"] == door].sort_values("time")

        for _, row in door_data.iterrows():
            marker = "o" if row["is_open"] else "x"
            color = "green" if row["is_open"] else "red"

            plt.scatter(row["time"], i, marker=marker, color=color, s=80)

    plt.yticks(range(len(doors)), doors)
    plt.xlabel("Time")
    plt.ylabel("Door")
    plt.title("Door State Change Timeline")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()


def add_any_door_open_column(steps):
    any_open = []

    for value in steps["door_states"]:
        states = parse_door_states(value)
        any_open.append(any(states.values()) if states else False)

    steps = steps.copy()
    steps["any_door_open"] = any_open
    return steps


def summarize_open_vs_closed(steps):
    steps = add_any_door_open_column(steps)

    summary = steps.groupby("any_door_open").agg(
        rows=("frame_count", "count"),
        avg_coverage=("coverage", "mean"),
        final_coverage=("coverage", "max"),
        avg_error=("error", "mean"),
        peak_error=("error", "max"),
        avg_visible_landmarks=("visible_landmarks", "mean"),
        total_collisions=("collision_count", "max"),
    )

    print("\nPerformance summary: doors closed vs doors open")
    print(summary)

    return summary


def plot_open_vs_closed_summary(steps):
    steps = add_any_door_open_column(steps)

    labels = {
        False: "All doors closed",
        True: "At least one door open"
    }

    grouped = steps.groupby("any_door_open").agg(
        avg_error=("error", "mean"),
        avg_coverage=("coverage", "mean"),
        avg_visible_landmarks=("visible_landmarks", "mean"),
    )

    grouped.index = [labels[i] for i in grouped.index]

    grouped.plot(kind="bar", figsize=(10, 6))
    plt.title("Performance During Closed vs Open Door Periods")
    plt.ylabel("Value")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def main():
    steps, events = load_data()

    print("Loaded steps:", len(steps))
    print("Loaded events:", len(events))

    plot_coverage_with_door_periods(steps, events)
    plot_error_with_door_events(steps, events)
    plot_visible_landmarks(steps, events)
    plot_robot_trajectory(steps)
    plot_door_timeline(events)

    summarize_open_vs_closed(steps)
    plot_open_vs_closed_summary(steps)


if __name__ == "__main__":
    main()