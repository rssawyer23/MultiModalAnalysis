"""Script for adding game score to event sequence and activity summary
        Using Rowe 2011 IJAIED article: Integrating Learning, Problem Solving, and Engagement in Narrative-centered Learning Environments"""
import pandas as pd
import numpy as np
import datetime
import csv


def get_student_solutions(activity_summary_filename):
    """Returns dataframe with index of TestSubject and the necessary columns of the student's solution"""
    act_sum = pd.read_csv(activity_summary_filename)
    act_sum = act_sum.loc[:, ["TestSubject", "SolutionDisease", "InfectionType", "SolutionObject", "SolutionTreatment"]]
    act_sum.index = act_sum["TestSubject"]
    act_sum.drop(labels=["TestSubject"], axis=1, inplace=True)
    return act_sum


def _award_worksheet(event_row):
    """Determine game score awarded for submitting a worksheet"""
    if "Wrong" in event_row["WorksheetSubmitResult"]:
        game_score_awarded = -100.0
    elif "Right" in event_row["WorksheetSubmitResult"]:
        game_score_awarded = 500.0
        game_score_awarded += 7500.0 / (float(event_row["GameTime"]) / 60.0)
    else:
        game_score_awarded = 0
        print("Unrecognized WorksheetSubmitResult: %s" % event_row["WorksheetSubmitResult"])
    return game_score_awarded


def _award_books(event_row, books):
    """Determine the game score awarded for reading a book and performing concept matrix attempt"""
    book_row = np.logical_and(books.loc[:, "TestSubject"] == event_row["TestSubject"],
                              books.loc[:, "GameTime"].apply(round, ndigits=2) == round(event_row["GameTime"],2))
    if np.sum(book_row) != 1:
        print("Error finding %s book at time %s" % (event_row["TestSubject"], round(event_row["GameTime"],2)))
        return 0.0
    concept_attempts = books.loc[book_row, "ConceptMatrixAttempts"].iloc[0]

    game_score_awarded = 0.0
    if concept_attempts == 1:
        game_score_awarded = 25.0
    elif concept_attempts == 2:
        game_score_awarded = 10.0
    elif concept_attempts == 3:
        game_score_awarded = -10.0
    return game_score_awarded


def _contaminant_comparison(event_type, solution_type):
    """Phrasing in Event Sequence different from phrasing in Activity Summary"""
    return event_type == "Viruses" and solution_type == "Viral" or \
           event_type == "Bacteria" and solution_type == "Bacterial"


def _award_scan(event_row, solutions):
    """Determining how to award points for a scanned object based on what correct solution is"""
    solution_row = solutions.loc[event_row["TestSubject"], :]
    correct_object = event_row["ObjectScanned"] == solution_row["SolutionObject"]
    correct_contaminant = _contaminant_comparison(event_row["TestingFor"], solution_row["InfectionType"])

    # Trying to mimic scheme from Rowe, but multiple solutions exist here
    #   Following scheme awards 200 for correct object and correct contaminant (same as Rowe)
    #                    awards 15  for correct contaminant but wrong object
    #                    awards -15 for correct object but wrong contaminant
    #                    awards -35 for wrong object and wrong contaminant (same as Rowe)
    game_score_awarded = 0.0

    # Awards
    if correct_object:
        game_score_awarded += 10.0
    if correct_contaminant:
        game_score_awarded += 25.0
    if correct_object and correct_contaminant:
        game_score_awarded += 165.0

    # Penalties
    if not correct_object:
        game_score_awarded -= 10.0
    if not correct_contaminant:
        game_score_awarded -= 25.0

    return game_score_awarded


def _award_conversation(event_row):
    game_score_awarded = 0.0
    elapsed_time = float(event_row["GameTime"]) / 60.0
    target = event_row["NPC"]
    if target == "Kim":
        game_score_awarded = 25.0
    elif target == "Teresa":
        game_score_awarded = 50.0
    elif target == "Ford" or target == "Robert" or target == "Quentin":
        game_score_awarded = 125.0
    return game_score_awarded / elapsed_time


def _event_game_score(event_row, solutions, books):
    """Determine the game score awarded for a specific event in Crystal Island"""
    game_score_awarded = 0.0
    if event_row["Event"] == "WorksheetSubmit":
        game_score_awarded = _award_worksheet(event_row)
    elif event_row["Event"] == "BooksAndArticles":
        game_score_awarded = _award_books(event_row, books)
    elif event_row["Event"] == "Scanner":
        game_score_awarded = _award_scan(event_row, solutions)
    elif event_row["Event"] == "Conversation":
        game_score_awarded = _award_conversation(event_row)
    return game_score_awarded


def add_game_score(event_filename, activity_summary, books_articles, appended_event_filename, appended_activity_summary):
    start_time = datetime.datetime.now()
    print("Started adding game score: %s" % start_time)
    student_solutions = get_student_solutions(activity_summary)
    books = pd.read_csv(books_articles)
    events = pd.read_csv(event_filename)
    events["GameScoreAwarded"] = events.apply(func=_event_game_score, axis=1,
                                               solutions=student_solutions, books=books)

    events["CumulativeGameScore"] = np.zeros(events.shape[0])
    subject_cumgs = pd.DataFrame(index=["TestSubject", "FinalGameScore"])
    for ts in list(events["TestSubject"].unique()):
        ts_rows = events.loc[:, "TestSubject"] == ts
        events.loc[ts_rows, "CumulativeGameScore"] = np.cumsum(events.loc[ts_rows, "GameScoreAwarded"])

        final_game_score = events.loc[ts_rows, "CumulativeGameScore"].iloc[-1]
        subject_cumrow = pd.Series({"TestSubject": ts, "FinalGameScore": final_game_score})
        subject_cumgs = pd.concat([subject_cumgs, subject_cumrow], axis=1)
    events.to_csv(appended_event_filename, index=False, doublequote=False)

    subject_cumgs = subject_cumgs.transpose()
    subject_cumgs.index = list(range(subject_cumgs.shape[0]))
    full_act_sum = pd.read_csv(activity_summary)
    try:
        full_act_sum = full_act_sum.drop(labels="VideoGamesPlayed", axis=1)
    except ValueError:
        pass
    full_act_sum = full_act_sum.merge(right=subject_cumgs, how="left", on="TestSubject")
    full_act_sum.to_csv(appended_activity_summary, index=False)
    finish_time = datetime.datetime.now()
    print("Finished adding game score in %.3f minutes" % ((finish_time - start_time).total_seconds() / 60.0))


if __name__ == "__main__":
    event_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/EventSequence/EventSequenceP.csv"
    activity_summary = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummary.csv"
    books_articles = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/BooksAndArticles/BooksAndArticles.csv"
    appended_event_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/EventSequence/EventSequence_wGS.csv"
    appended_activity_summary = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/ActivitySummary/ActivitySummary_wGS.csv"
    add_game_score(event_filename, activity_summary, books_articles, appended_event_filename, appended_activity_summary)
