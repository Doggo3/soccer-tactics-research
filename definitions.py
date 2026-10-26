FIELDS = {
    "ball_x_start": (
        "Ball starting x position",
        "m",
        "Position along the pitch at the start of the phase. Zero is midfield. These are the saved tracking coordinates; the app does not infer attacking direction.",
    ),
    "ball_progress_m": (
        "Signed ball progress",
        "m",
        "Change in the ball’s x coordinate over the phase. A negative value means movement toward decreasing x, not automatically a backward attack.",
    ),
    "duration": ("Phase duration", "s", "How long this section of play lasted."),
    "progress_speed_mps": (
        "Signed progress speed",
        "m/s",
        "Signed x progress divided by duration. This is not the ball’s total running speed.",
    ),
    "attack_width_mean": (
        "Attacking team width",
        "m",
        "Average side-to-side spread of the attacking team.",
    ),
    "attack_depth_mean": (
        "Attacking team depth",
        "m",
        "Average front-to-back spread of the attacking team.",
    ),
    "attack_spacing_mean": (
        "Attacking player spacing",
        "m",
        "Phase-average spacing measurement for attacking players.",
    ),
    "defense_width_mean": (
        "Defending team width",
        "m",
        "Average side-to-side spread of the defending team.",
    ),
    "defense_depth_mean": (
        "Defending team depth",
        "m",
        "Average front-to-back spread of the defending team.",
    ),
    "defense_spacing_mean": (
        "Defending player spacing",
        "m",
        "Phase-average spacing measurement for defending players.",
    ),
    "defenders_within_5m_mean": (
        "Defenders within 5 m",
        "players",
        "Average number of defenders within 5 meters of the ball.",
    ),
    "defenders_within_10m_mean": (
        "Defenders within 10 m",
        "players",
        "Average number of defenders within 10 meters of the ball.",
    ),
    "attackers_ahead_of_ball_mean": (
        "Attackers ahead of the ball",
        "players",
        "Average count across the phase; fractional values are possible.",
    ),
    "attackers_in_wide_zones_mean": (
        "Attackers in wide zones",
        "players",
        "Average count of attackers in the dataset’s wide zones.",
    ),
    "ball_wide_share": (
        "Ball wide share",
        "share",
        "Fraction of sampled observations with the ball in the dataset’s wide zones.",
    ),
    "frames_sampled": (
        "Sampled frames",
        "frames",
        "Number of tracking observations used to summarize the phase.",
    ),
    "match_minute": (
        "Minutes elapsed in half",
        "min",
        "Elapsed time within the recorded half, restarting at zero for the second half. This is not the full-match clock.",
    ),
    "period": ("Match half", "half", "1 = first half; 2 = second half."),
}
