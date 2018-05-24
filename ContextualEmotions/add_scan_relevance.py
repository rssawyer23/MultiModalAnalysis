import pandas as pd


def detect_positive_scan(row, solution, object):
    correct_solution = row["TestingFor"] == "Bacteria" and solution == "Bacterial" or row["TestingFor"] == "Viruses" and solution == "Viral"
    correct_object = row["ObjectScanned"] == object
    return correct_solution and correct_object


def add_scan_relevancies(scans_filename, act_df_filename, output_filename=""):
    """Add the positive and negative scan counts to the activity summary
        If output_filename is empty string, will overwrite act_df_filename"""
    act_df = pd.read_csv(act_df_filename)
    act_df.index = act_df["TestSubject"]
    act_df["ScanPositive"] = pd.Series([0] * act_df.shape[0])
    act_df["ScanNegative"] = pd.Series([0] * act_df.shape[0])
    scan_df = pd.read_csv(scans_filename)

    subjects = list(act_df["TestSubject"].unique())
    for subject in subjects:
        subject_solution = act_df.loc[subject, "InfectionType"]
        subject_object = act_df.loc[subject, "SolutionObject"]

        scan_rows = scan_df.loc[scan_df["TestSubject"] == subject, :]
        scan_rows["Positive"] = scan_rows.apply(lambda row: detect_positive_scan(row, subject_solution, subject_object), axis=1)
        positive_scans = scan_rows["Positive"].sum()
        negative_scans = scan_rows.shape[0] - positive_scans
        act_df.loc[subject, ["ScanPositive", "ScanNegative"]] = pd.Series(data=[positive_scans, negative_scans], index=["ScanPositive", "ScanNegative"])

    if output_filename == "":
        act_df.to_csv(act_df_filename, index=False)
    else:
        act_df.to_csv(output_filename, index=False)


def add_book_relevancies(books_filename, act_df_filename, output_filename=""):
    act_df = pd.read_csv(act_df_filename)
    act_df.index = act_df["TestSubject"]
    act_df["BookRelevant"] = pd.Series([0] * act_df.shape[0])
    act_df["BookIrrelevant"] = pd.Series([0] * act_df.shape[0])
    book_df = pd.read_csv(books_filename)

    subjects = list(act_df["TestSubject"].unique())
    for subject in subjects:
        book_rows = book_df.loc[book_df["TestSubject"] == subject, :]
        relevant_books = book_rows.loc[:, "IsReleventToSolution"].sum()
        irrelevant_books = book_rows.shape[0] - relevant_books
        act_df.loc[subject, ["BookRelevant", "BookIrrelevant"]] = pd.Series(data=[relevant_books, irrelevant_books], index=["BookRelevant", "BookIrrelevant"])

    if output_filename == "":
        act_df.to_csv(act_df_filename, index=False)
    else:
        act_df.to_csv(output_filename, index=False)


def add_submit_relevancies(act_df_filename, output_filename=""):
    act_df = pd.read_csv(act_df_filename)
    act_df.index = act_df["TestSubject"]
    act_df["SubmissionCorrect"] = pd.Series(act_df["MysterySolved"], dtype=int)
    act_df["SubmissionIncorrect"] = act_df.loc[:, "TotalWorksheetSubmits"] - act_df.loc[:, "SubmissionCorrect"]

    if output_filename == "":
        act_df.to_csv(act_df_filename, index=False)
    else:
        act_df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    data_directory = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output318/"
    out_directory = "C:/Users/robsc/Documents/NC State/GRAWork/Publications/LI-Contextual-Emotions/Data/"

    scans_file = data_directory + "Scanner/Scanner.csv"
    books_file = data_directory + "BooksAndArticles/BooksAndArticles.csv"

    dest_act_sum = out_directory + "ActivitySummaryContextEdited.csv"

    add_scan_relevancies(scans_filename=scans_file,
                         act_df_filename=dest_act_sum)
    add_book_relevancies(books_filename=books_file,
                         act_df_filename=dest_act_sum)
    add_submit_relevancies(act_df_filename=dest_act_sum)
