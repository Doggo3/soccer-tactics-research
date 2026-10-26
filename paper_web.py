"""Readable web edition; research content condensed from the supplied manuscript."""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

P = Path(__file__).parent


def render(paper, chart, goto):
    info = json.loads((P / "paper_info.json").read_text())
    st.header(info["title"])
    st.write("**Jack Chiang**")
    st.caption("Web edition · Condensed from the supplied manuscript")
    part = st.radio(
        "Read a section",
        [
            "Abstract",
            "Introduction",
            "Data & methods",
            "Findings",
            "Conclusion",
            "References",
        ],
        horizontal=True,
        key="paper_section",
    )
    st.divider()
    if part == "Abstract":
        st.subheader("Abstract")
        st.write(info["abstract"])
        st.caption("Original abstract, with typography normalized.")
        st.button(
            "Continue to the introduction",
            on_click=lambda: st.session_state.update(paper_section="Introduction"),
        )
    elif part == "Introduction":
        st.subheader("From tracking measurements to tactical labels")
        st.write(
            "Soccer tactics are usually identified by coaches or analysts watching match footage. They consider player positioning, ball movement, and how a team reacts after gaining possession. Reviewing many matches manually takes time, and different analysts may describe the same attack differently."
        )
        st.write(
            "This study asks whether player spacing, team shape, ball position, and movement during a phase can identify the type of attack. The task is supervised multiclass classification: each recorded phase becomes a vector of numerical inputs, and the model chooses one of five labels."
        )
        st.markdown(
            "**Build-up** — a controlled attack, normally beginning farther from the opponent’s goal.\n\n**Creation** — developing an attacking opportunity.\n\n**Direct attack** — moving the ball forward quickly.\n\n**Fast transition** — attacking soon after gaining possession.\n\n**Finishing** — the final part of an attack near the opponent’s goal."
        )
        st.write(
            "The reference labels were supplied by SkillCorner’s Phases of Play system. The model therefore learns to reproduce that system’s labels from a smaller set of tracking features. This is different from independently validating the labels with coaches."
        )
        st.subheader("Why summarize a phase?")
        st.write(
            "Tracking data describes player and ball locations, providing information about team shape beyond individual events such as passes and shots. Summarizing those observations makes feature-based models easier to train and inspect. The trade-off is that averages discard the order of events—a potentially important distinction between a transition and an established attack."
        )
    elif part == "Data & methods":
        st.subheader("Dataset and evaluation")
        a, b, c = st.columns(3)
        a.metric("Matches", "10")
        b.metric("Phases", "2,955")
        c.metric("Inputs per phase", "18")
        st.write(
            "The study uses SkillCorner Open Data from Australian A-League matches. The original source includes tracking coordinates and match, player, possession, and phase information. The processed CSV contains one row per attacking phase."
        )
        split = pd.DataFrame(
            {"Split": ["Training", "Testing"], "Matches": [8, 2], "Phases": [2388, 567]}
        )
        st.dataframe(split, hide_index=True, width="stretch")
        st.write(
            "Training and testing use different matches. Test matches 2015213 and 1899585 are held out, so phases from the same game do not appear on both sides of this split."
        )
        st.subheader("What the model measures")
        st.markdown(
            "- **Ball movement:** starting position, signed progress, and time spent in wide areas.\n- **Team structure:** attacking and defending width, depth, and spacing.\n- **Local pressure:** defenders near the ball and attackers ahead or in wide zones.\n- **Sampling and timing:** sampled frames, half, elapsed minute, duration, and signed progress speed."
        )
        st.write(
            "The original configuration has 14 inputs. The enhanced configuration adds elapsed minute, half, phase duration, and progress speed. Match identifiers, phase identifiers, and the target label are excluded from the inputs."
        )
        st.subheader("Models and metrics")
        st.write(
            "The study compares Random Forest, XGBoost, SVC, MLP, and KNN. Models sensitive to feature scale use standardized inputs; tree-based models do not require scaling. Class or sample weighting is used when supported to account for unequal class sizes."
        )
        st.write(
            "Accuracy measures the fraction of all test labels predicted correctly. Macro F1 averages each category’s F1 score equally, combining precision and recall without letting the largest class dominate. Confusion matrices show the kinds of mistakes, and feature-importance experiments examine which measurements the models use."
        )
        st.button(
            "Explore the recorded measurements",
            on_click=goto,
            args=("Explore the data",),
        )
    elif part == "Findings":
        st.subheader("Strong overall accuracy, uneven performance by tactic")
        st.write(
            "Enhanced XGBoost reached 91.9% accuracy and 0.864 macro F1. Enhanced Random Forest reached 91.2% accuracy and the highest macro F1 in the reported comparison, 0.873. The highest overall accuracy and the most balanced category performance therefore came from different models."
        )
        labels = [
            "Build-up",
            "Creation",
            "Direct attack",
            "Fast transition",
            "Finishing",
        ]
        cm = paper["confusion"]
        counts = [sum(r) for r in cm]
        correct = [cm[i][i] for i in range(5)]
        frame = pd.DataFrame(
            {
                "Phase": labels,
                "Correct": [100 * a / b for a, b in zip(correct, counts)],
                "Identified": [f"{a} of {b}" for a, b in zip(correct, counts)],
            }
        )
        fig = px.bar(
            frame,
            x="Correct",
            y="Phase",
            orientation="h",
            text="Identified",
            range_x=[0, 100],
            color_discrete_sequence=["#28736b"],
            labels={"Correct": "Correctly identified (%)"},
        )
        fig.update_traces(textposition="inside")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        chart(fig, "web_paper_recall", 330)
        st.caption(
            "Figure 1. Recall by category, calculated from the saved notebook confusion matrix. Each denominator is the number of test phases with that reference label. This is recall, not F1."
        )
        st.write(
            "Fast transition was the hardest category: 13 of its 25 test examples were correctly identified. Most mistakes assigned creation or finishing. Transitions can share spatial characteristics with those phases, while averages lose the sequence of actions following a change of possession. The paper proposes this lost context and the small class size as possible explanations."
        )
        st.subheader("Which measurements were informative?")
        st.write(
            "Ball starting position had the largest XGBoost importance score (0.1961), followed by sampled frames (0.1187), ball progress (0.1124), and attackers ahead of the ball (0.1018). These are model-level importance scores, not proof that one feature caused a particular prediction."
        )
        st.write(
            "Removing groups of weaker features produced small changes in performance, without establishing that a smaller feature set was reliably better. Match-based validation also showed variation across held-out groups, reinforcing the need to evaluate on more than one split."
        )
        st.button("Inspect the full results", on_click=goto, args=("Results & method",))
    elif part == "Conclusion":
        st.subheader("What the study supports")
        st.write(
            "Numerical summaries of player and ball tracking can reproduce the five supplied attacking-phase labels with strong overall accuracy in these matches. Random Forest and XGBoost produced the strongest reported results, while fast transition remained difficult."
        )
        st.subheader("What remains unresolved")
        st.write(
            "The study includes only 10 matches from one league and uses labels generated by SkillCorner’s system. It does not establish performance across competitions or agreement with independent coaches. Phase averages make the models manageable but remove temporal information that may distinguish similar attacks."
        )
        st.subheader("Next steps")
        st.write(
            "The paper proposes collecting more matches from different teams and leagues, examining individual correct and incorrect predictions, and adding context such as score difference and the preceding tactic. Sequence models could preserve the order of player and ball movements rather than reducing a phase to averages."
        )
        st.subheader("Acknowledgements")
        st.write(
            "The author thanks Omar Ramos Escoto for research guidance and feedback, Inspirit AI for supporting the project, and SkillCorner for providing open tracking data."
        )
    elif part == "References":
        st.subheader("References in the manuscript")
        st.markdown(
            """1. SkillCorner. [SkillCorner Open Data](https://github.com/SkillCorner/opendata).
2. SkillCorner. Phases of Play CSV Specifications.
3. Y. Li and D. Link. [Intention Driven Identification of In-Possession Match Phases in Association Football Through Temporal Graph Learning](https://arxiv.org/abs/2606.09289).
4. [An AI Framework for Counterattack Detection and Decision-Making Evaluation in Football](https://link.springer.com/article/10.1186/s40537-025-01128-3).
5. [Estimation of Overlapped Tactical Actions from Soccer Tracking Data](https://www.scitepress.org/Papers/2024/130663/130663.pdf).
6. [Mapping Football Tactical Behavior and Collective Dynamics with Artificial Intelligence: A Systematic Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12163489/)."""
        )
        st.caption("References transcribed from the supplied manuscript.")
    with st.expander("Citation"):
        st.write(
            info["author"] + ". " + info["title"] + ". Author-provided manuscript."
        )
        st.code(
            "@unpublished{chiang_soccer_attacking_phases,\n  author = {Chiang, Jack},\n  title = {"
            + info["title"]
            + "},\n  note = {Author-provided manuscript},\n  url = {https://machine-learning-soccer-attacking-phase-classification.streamlit.app/?section=paper}\n}",
            language="bibtex",
        )
