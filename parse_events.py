# Script for parsing the EventSequenceFACET.csv file into supervised learning samples
# Revised for the UMUAI Crystal Island modelling

import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse

# Dictionary for each category of duration proportions.
# Each sub-category corresponds to a proportion in relation to the total time
# in which the student was fixated on a target.
DURATION_TARGET_DICT = {"NPCsDuration": ["KimDuration", "QuentinDuration", "FordDuration", "RobertDuration", "TeresaDuration",
                                 "SamDuration", "GregDuration", "BryceDuration"],
                        "Travel/GameItemsDuration": ["TerrainDuration", "BoatDuration", "DockDuration",
                                                     "BeachHutDuration", "EliseBeachDuration","WindTurbineDuration",
                                                     "FirePitDuration", "OrchidPlantDuration", "BackpackDuration",
                                                     "TableBesideWindTurbineDuration", "LeafyPlantDuration",
                                                     "RocksDuration", "FirePitTableBookDuration", "Chair1Duration",
                                                     "InfirmaryDuration", "InfirmaryDoorDuration", "IslandDuration",
                                                     "DarkWoodDuration", "BryceQuartersDuration",
                                                     "LivingQuartersDoor2Duration", "DormDuration",
                                                     "DormFurnitureDuration", "MesquitoNet1Duration",
                                                     "MesquitoNet2Duration", "InfirmaryFurnitureDuration",
                                                     "InfluenzaPosterDuration", "InfirmaryStandArticleDuration",
                                                     "TrophiesDuration", "LivingQuartersArticleDuration",
                                                     "Bench3Duration", "DiningHallWallArticleDuration",
                                                     "Bench1Duration", "BryceCollectibleDuration",
                                                     "BryceEntryUpperShelfBookDuration",
                                                     "BryceEntryLowerShelfBookDuration",
                                                     "BryceQuartersFurnitureDuration", "BryceEntrySofaDuration",
                                                     "BryceQuartersDoorDuration", "InfirmaryDoorWindowDuration",
                                                     "LivingQuartersFloorArticleDuration", "FirePitTableDuration",
                                                     "LivingQuartersDoor1Duration", "DiningHallCounterArticleDuration",
                                                     "BryceBedsideArticleDuration", "ShelvesDuration", "PlantsDuration",
                                                     "StairsDuration", "DiningHallFloorArticleDuration",
                                                     "MesquitoNet3Duration", "LivingQuartersDoor4Duration",
                                                     "SkeletonKeyDuration", "ChairDuration", "LightFixturesDuration",
                                                     "LivingQuartersDoor3Duration", "MagnifyingGlassDuration",
                                                     "lightDuration", "TreadsDuration", "BulbsDuration",
                                                     "light1Duration", "BenchDuration",
                                                     "Bryce_House_Stair_Rail_2Duration", "fanBlades1Duration",
                                                     "fanBlades2Duration"],
                        "Food_RelatedDuration": ["DiningHallDuration", "DiningHallDoorDuration", "AppleDuration",
                                                 "BreadDuration", "BananaDuration", "CoconutDuration",
                                                 "DiningHallCabinetsAndStoolsDuration", "Glass1Duration",
                                                 "DiningHallTablesDuration", "ChickenLegDuration", "PieDuration",
                                                 "stoveDuration", "MilkDuration", "FridgeDoorDuration", "EggDuration",
                                                 "OrangeJuiceDuration", "CheeseDuration", "OrangeDuration",
                                                 "FridgeDuration", "DiningHallRoofDuration", "SandwichDuration",
                                                 "ChickenRawDuration", "PeanutsDuration", "TomatoDuration",
                                                 "potDuration", "WaterDuration", "LettuceLeafDuration",
                                                 "JellyDuration", "YogurtDuration", "waterDuration"],
                        "Lab_RelatedDuration": ["LabDuration", "LabDoorDuration", "LabEquipmentDuration",
                                                "Elise LabDuration", "ScanningDeviceDuration", "ScannerDuration",
                                                "LabPapersOnCounterDuration", "polySurface6Duration"],
                        "Diagnosis_RelatedDuration": ["DiagnosisWorksheetDuration", "SalmonellosisPosterDuration",
                                                      "SmallpoxPosterDuration", "BotulismPosterDuration",
                                                      "BacterialReproductionPosterDuration",
                                                      "VirusStructurePosterDuration", "VirusReproductionPosterDuration",
                                                      "SizeComparisonPosterDuration", "EbolaPosterDuration",
                                                      "BacteriaStructurePosterDuration"],
                        "Miscellaneous/MessagesDuration": ["WaitIndicatorDuration", "HudDuration",
                                                           "DialogContentDuration",
                                                           "LostInvestigationAnimationDuration", "EndGameDuration",
                                                           "MessageBoxDuration", "PauseMenuDuration",
                                                           "SettingsDuration", "FastTravelDuration",
                                                           "AchievementsDuration", "TutorialMessageDuration"],
                        "Book_RelatedDuration": ["BookContentDuration", "DiningHallTableBookDuration",
                                                 "ResearchArticleContentDuration", "InfirmaryMidShelfBookDuration",
                                                 "InfirmaryUpperShelfBookDuration",
                                                 "LivingQuartersEndTableBookDuration",
                                                 "LivingQuartersDresserBookDuration",
                                                 "LivingQuartersBesideChairBookDuration",
                                                 "DiningHallCounterBookDuration", "BryceOfficeBookDuration",
                                                 "BryceOfficeBookTopShelfDuration", "LabUpperShelfBookDuration"],
                        "Concept_Matrix_RelatedDuration": ["ConceptMatrixDuration", "ConceptMatrixFeedbackDuration",
                                                           "ConceptMatrixFolderDuration"]}
FEATURES_DURATION = ["NPCsDuration", "Travel/GameItemsDuration", "Food_RelatedDuration", "Lab_RelatedDuration",
                     "Diagnosis_RelatedDuration", "Miscellaneous/MessagesDuration", "Book_RelatedDuration",
                     "Concept_Matrix_RelatedDuration", "Fixations/Sec", "Game Time"]
ACTIVITY_FEATURES = ["PreTestScore", "Pre-AGQ-Mastery-Approach", "Pre-AGQ-Performance-Approach",
                     "Pre-AGQ-Mastery-Avoidance", "Pre-AGQ-Performance-Avoidance", "TestSubject"]


def create_subject_event_blocks(event_df, omit_list):
    """Function that returns a dictionary of TestSubject:EventDataFrame pairs based on full event dataframe"""
    subjects = [e for e in list(event_df["TestSubject"].unique()) if e not in omit_list]
    df_blocks = dict()
    for subject in subjects:
        subject_rows = event_df["TestSubject"] == subject
        subject_df = event_df.loc[subject_rows, :]
        subject_df.index = list(range(subject_df.shape[0]))
        df_blocks[subject] = subject_df.copy()
    return df_blocks


def generate_time_series_indices(time_stamps, total_seconds, interval=10):
    """Passing the TimeStamps of a student's events dataframe and total seconds, generate sequence indices for distance calculations"""
    start_time = parse(time_stamps.iloc[0])
    time_intervals = [start_time + datetime.timedelta(seconds=i*interval) for i in range(int(total_seconds / interval))]
    sequence_indices = []
    sequence_counter = 0
    for i in range(len(time_intervals)):
        while sequence_counter + 1 < len(time_stamps) and parse(time_stamps.iloc[sequence_counter + 1]) < time_intervals[i]:
            sequence_counter += 1
        sequence_indices.append(sequence_counter)
    sequence_indices.append(len(time_stamps)-1)
    return sequence_indices


def cumulative_event_blocks(event_filename, activity_filename, interval, add_columns, keep_list):
    """Function for parsing the EventSequenceFACET.csv file which already has cumulatives appended for each subject
    This function acts as the main for the affect student modeling of """
    event_df = pd.read_csv(event_filename)
    act_df = pd.read_csv(activity_filename)
    omit_list = [e for e in list(event_df["TestSubject"].unique()) if "1301" not in e]  # removing partial agency students
    subject_blocks = create_subject_event_blocks(event_df, omit_list=omit_list)
    subject_counts = pd.Series()
    supervised_examples = pd.DataFrame()

    for subject in subject_blocks.keys():
        if subject in keep_list:
            add_student_data = get_student_data(act_df, subject, add_list=add_columns)
            start_time = parse(subject_blocks[subject].loc[:, "TimeStamp"].iloc[0])
            subject_blocks[subject].loc[:, "GameTime"] = subject_blocks[subject].loc[:, "TimeStamp"].apply(lambda cell: (parse(cell) - start_time).total_seconds())
            parsed_subject_block = create_rows(subject_blocks[subject], parsing_method=interval)
            parsed_subject_block.index = list(range(parsed_subject_block.shape[0]))
            subject_counts[subject] = parsed_subject_block.shape[0]
            for act_feat in list(add_student_data.index):
                parsed_subject_block[act_feat] = pd.Series([add_student_data[act_feat]] * parsed_subject_block.shape[0])
            parsed_subject_block["IntervalID"] = pd.Series([e+1 for e in range(parsed_subject_block.shape[0])])
            supervised_examples = pd.concat([supervised_examples, parsed_subject_block], axis=0)
    supervised_examples.index = list(range(supervised_examples.shape[0]))
    return supervised_examples, subject_counts


def create_rows(subject_event_df, parsing_method):
    """Create rows of events by using indices from parsing method
    Works similar to create_blocks, except now returning one block which are selected rows from a larger event dataframe"""
    if parsing_method >= 1:  # parse using seconds
        second_interval = int(parsing_method)
        start_time = parse(subject_event_df["TimeStamp"].iloc[0])
        total_seconds = (parse(subject_event_df["TimeStamp"].iloc[-1]) - start_time).total_seconds()
        sequence_indices = generate_time_series_indices(subject_event_df["TimeStamp"], total_seconds=total_seconds, interval=second_interval)
        block = subject_event_df.iloc[sequence_indices, :]
    elif 0 <= parsing_method < 1:
        portion = min(0.5, parsing_method)
        indices_per_frame = (subject_event_df.shape[0] * portion)
        frames = int(1.0 / portion)
        sequence_indices = []
        for frame_num in range(frames):
            start_index = int(frame_num * indices_per_frame)
            end_index = int((frame_num + 1) * indices_per_frame)
            sequence_indices.append(end_index)
        block = subject_event_df.iloc[sequence_indices, :]
    else:
        print("Undefined parsing method: %s" % parsing_method)
        block = pd.DataFrame()
    return block


def create_blocks(subject_event_df, parsing_method):
    """Create blocks of events by blocking event_df using parsing defined by parsing_method
            parsing_method: if >= 1, uses every (parsing_method) seconds to determine block length
                            if 0 <= parsing_method < 1, uses (parsing_method) percent of events for blocks"""
    all_blocks = []
    if parsing_method >= 1:  # parse using seconds
        second_interval = int(parsing_method)
        start_time = parse(subject_event_df["TimeStamp"].iloc[0])
        total_seconds = (parse(subject_event_df["TimeStamp"].iloc[-1]) - start_time).total_seconds()
        sequence_indices = generate_time_series_indices(subject_event_df["TimeStamp"], total_seconds=total_seconds, interval=second_interval)
        for i in range(1, len(sequence_indices)):
            block = subject_event_df.iloc[sequence_indices[i-1]:sequence_indices[i], :]
            all_blocks.append(block.copy())
    elif 0 <= parsing_method < 1:
        portion = min(0.5, parsing_method)
        indices_per_frame = (subject_event_df.shape[0] * portion)
        frames = int(1.0 / portion)
        for frame_num in range(frames):
            start_index = int(frame_num * indices_per_frame)
            end_index = int((frame_num+1) * indices_per_frame)
            block = subject_event_df.iloc[start_index:end_index, :]
            all_blocks.append(block.copy())
    else:
        print("Undefined parsing method: %s" % parsing_method)
    return all_blocks


def initialize_feature_series(add_data):
    """Creates a series that will be the X of the supervised learning example, depends on globally defined variables"""
    series = add_data.copy()
    for key in FEATURES_DURATION:
        series[key] = 0.0
    return series


def _convert_target_category(aoi_target):
    """Convert the fine-grained aoi target to an interpretable category"""
    category = "UNIDENTIFIED AOI_TARGET"
    for key in DURATION_TARGET_DICT.keys():
        if aoi_target+"Duration" in DURATION_TARGET_DICT[key]:
            category = key
    return category


def condense_blocks_to_aoi_series(subject_blocks, cumulative, difference, add_data):
    """Taking blocks from a subject, condense them into a series representing the X of a supervised learning sample X -> y
            cumulative: Boolean determining whether blocks should be treated individually or summed up to current block
            difference: Boolean determining whether labels should be difference between start/end game score or the end game score"""
    cumulative_series = initialize_feature_series(add_data)
    sum_keys = [e for e in list(cumulative_series.index) if e not in list(add_data.index)]
    examples_dataframe = pd.DataFrame()
    labels = []
    aoi_keys = list(DURATION_TARGET_DICT.keys())
    for block in subject_blocks:
        if block.shape[0] > 0:
            # Condense the block into a series of (summation) features
            feature_series = initialize_feature_series(add_data)  # Depends on globally defined variables
            feature_series.loc["Game Time"] = (parse(block["TimeStamp"].iloc[-1]) - parse(block["TimeStamp"].iloc[0])).total_seconds()
            for i,row in block.iterrows():
                if row["Event"] == "AOI":
                    aoi_key = _convert_target_category(row["Target"])
                    if aoi_key != "UNIDENTIFIED AOI_TARGET":
                        feature_series[aoi_key] += row["Duration"]
                        feature_series["Fixations/Sec"] += 1
            cumulative_series.loc[sum_keys] += feature_series.loc[sum_keys]
            cumulative_series.loc["Game Time"] += feature_series.loc["Game Time"]

            # Get the game score label for the block with either difference or last
            try:
                if difference:
                    label = block.loc[:, "CumulativeGameScore"].iloc[-1] - block.loc[:, "CumulativeGameScore"].iloc[0]
                else:
                    label = block.loc[:, "CumulativeGameScore"].iloc[-1]
            except IndexError:
                print(block)
                label = 0.0
            labels.append(label)

            # Calculate proportions and append to dataframe using either cumulative or block specific features
            fixation_sum = np.sum(feature_series.loc[aoi_keys])
            if cumulative and fixation_sum > 0:
                proportion_cumulative_series = cumulative_series.copy()
                proportion_cumulative_series.loc[aoi_keys] = cumulative_series.loc[aoi_keys] / np.sum(cumulative_series.loc[aoi_keys])
                proportion_cumulative_series.loc["Fixations/Sec"] = proportion_cumulative_series.loc["Fixations/Sec"] / proportion_cumulative_series.loc["Game Time"]
                examples_dataframe = pd.concat([examples_dataframe, proportion_cumulative_series.copy()], axis=1)
            elif fixation_sum > 0:
                proportion_feature_series = feature_series.copy()
                proportion_feature_series.loc[aoi_keys] = feature_series.loc[aoi_keys] / np.sum(feature_series.loc[aoi_keys])
                proportion_feature_series.loc["Fixations/Sec"] = proportion_feature_series.loc["Fixations/Sec"] / proportion_feature_series.loc["Game Time"]
                examples_dataframe = pd.concat([examples_dataframe, proportion_feature_series.copy()], axis=1)

    # Adding labels as a series in the dataframe
    examples_dataframe = examples_dataframe.T
    examples_dataframe.index = list(range(examples_dataframe.shape[0]))
    examples_dataframe["Label"] = pd.Series(labels)
    return examples_dataframe


def get_student_data(act_df, subject, add_list):
    """Function for getting additional student data from the ActivitySummary (i.e. graded surveys)"""
    subject_rows = act_df.loc[:, "TestSubject"] == subject
    if np.sum(subject_rows) != 1:
        print("Error with getting data for %s" % subject)
        return pd.Series()
    else:
        subject_data = act_df.loc[subject_rows, add_list]
        return subject_data.squeeze()


if __name__ == "__main__":
    # Arguments of the later functions
    event_filepath = "C:/Users/Andrew/Documents/NC State Graduate Program/CEI/Crystal Island Data/EventSequence_wCGS_AOI.csv"
    act_filepath = "C:/Users/Andrew/Documents/NC State Graduate Program/CEI/Crystal Island Data/ActivitySummaryGraded.csv"
    output_filepath = "C:/Users/Andrew/Documents/NC State Graduate Program/CEI/Crystal Island Data/CumulativeScoring(20).csv"
    omit_list = []
    parsing_method = 0.20
    cumulative = True  # Cumulative and use_gs_difference should probably be opposites
    use_gs_difference = False
    show = True

    print("Starting reading data into subject blocks: %s" % datetime.datetime.now())
    event_df = pd.read_csv(event_filepath)
    act_df = pd.read_csv(act_filepath)
    omit_list = [e for e in list(event_df["TestSubject"].unique()) if "1301" not in e]  # removing partial agency students
    subject_blocks = create_subject_event_blocks(event_df, omit_list=omit_list)
    subject_counts = pd.Series()
    supervised_examples = pd.DataFrame()

    print("Starting parsing each subject block %s" % datetime.datetime.now())
    for subject in subject_blocks.keys():
        add_student_data = get_student_data(act_df, subject, add_list=ACTIVITY_FEATURES)
        parsed_subject_blocks = create_blocks(subject_blocks[subject], parsing_method=parsing_method)
        subject_examples = condense_blocks_to_aoi_series(parsed_subject_blocks, cumulative=cumulative, difference=use_gs_difference, add_data=add_student_data)
        subject_counts[subject] = len(parsed_subject_blocks)  # If parsing method is a percent, should be equal for all subjects
        supervised_examples = pd.concat([supervised_examples, subject_examples], axis=0)
        if show:
            print("Finished parsing %s at %s" % (subject, datetime.datetime.now()))

    # print(subject_counts.sort_values())  # For confirming correct handling of parsing and invalid students
    supervised_examples.to_csv(output_filepath, index=False)
