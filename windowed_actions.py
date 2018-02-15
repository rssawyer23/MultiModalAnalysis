from datetime import datetime, timezone
from datetime import date
from datetime import timedelta
from dateutil import parser
from sklearn.preprocessing import scale
from facet_event_file import index_dictionary
import numpy as np
import pandas as pd
# event_facet_filename is the merged event sequence with facet events

ACTION_TYPES = {"KnowledgeAcquisition": ["BooksAndArticles", "PostersLookedAt"],
                "InformationGathering": ["Conversation"],
                "HypothesisTesting": ["Worksheet", "WorksheetSubmit", "Scanner"]}


# Data structure for organizing a line (event) into its relevant time and identifying properties
class data_line:
    def __init__(self, line, ind, window):
        line_split = line.replace("\n","").replace("\r","").split(",")
        if len(line_split) < 5:
            self.line_type = "NULL"
        else:
            self.test_subject = line_split[ind["TestSubject"]]
            self.line_type = line_split[ind["Event"]]
            try:
                self.duration = float(line_split[ind["Duration"]])
            except ValueError:
                self.duration = 0
            self.start_time = parser.parse(line_split[ind["TimeStamp"]])
            self.before_start = self.start_time - timedelta(seconds=window)
            self.end_during = self.start_time + timedelta(seconds=self.duration)
            self.end_after = self.end_during + timedelta(seconds=window)
            self.target = line_split[ind["Target"]]


# Data structure for keeping a list of lines and maintenance functions
class data_list:
    def __init__(self):
        self.list = []

    # Used on the action list when a new data line is read
    # determines which actions have "expired" or cannot have new elements in their after phase given the most recent timestamp observed
    def return_expired_actions(self, dl):
        actions_to_write = []
        i = 0
        while i < len(self.list):
            if dl.start_time > self.list[i].end_after:
                actions_to_write.append(self.list.pop(i)) # popping shortens list by one so no increment to index/pointer
            else:
                i += 1

        return actions_to_write

    # Used on the action list to assist the facet list maintenance
    # The earliest timestamp here is the latest timestamp that needs to be kept on the facet list
    def return_earliest_before_ts(self):
        if len(self.list) > 0:
            return self.list[0].before_start
        else:
            return datetime.now(timezone.utc)

    def add_data(self, dl):
        self.list.append(dl)

    # Used on the facet list to remove facet events that cannot be in any actions intervals
    # Done by checking facet events last possible moment of relevance against the before interval of earliest action)
    def trim(self, earliest_timestamp):
        i = 0
        while i < len(self.list):
            if earliest_timestamp > self.list[i].end_after:
                self.list.pop(i)  # popping shortens list by one so no increment to index/pointer
            else:
                i += 1


class student_data:
    def __init__(self, dl):
        self.id = dl.test_subject
        self.data = dict()

    def write_student(self, ofile, ordered_actions, ordered_emotions):
        output_string = "%s," % self.id
        for a in ordered_actions:
            try:
                output_string += self.data[a].get_action_string(ordered_emotions)
            except KeyError:
                output_string += action_data(a).get_action_string(ordered_emotions)

        output_string = output_string[:-1] + "\n"
        ofile.write(output_string)

    def add_action_data(self, action_type, facet_list):
        if action_type.line_type not in self.data.keys():
            self.data[action_type.line_type] = action_data(action_type.line_type)
        for f in facet_list.list:
            if action_type.before_start < f.end_during < action_type.start_time or action_type.before_start < f.start_time < action_type.start_time:
                self.data[action_type.line_type].increment_affect("Before", f.target, f.duration)
            if action_type.start_time < f.end_during < action_type.end_during or action_type.start_time < f.start_time < action_type.end_during:
                self.data[action_type.line_type].increment_affect("During", f.target, f.duration)
            if action_type.end_during < f.end_during < action_type.end_after or action_type.end_during < f.start_time < action_type.end_after:
                self.data[action_type.line_type].increment_affect("After", f.target, f.duration)
        self.data[action_type.line_type].increment(action_type.duration)


# Data structure for keeping track of emotions for each action
# Before, During, After intervals around each action
class action_data:
    def __init__(self, name):
        self.name = name
        self.occurrences = 0
        self.duration = 0
        self.data = dict()
        self.data["Before"] = dict()
        self.data["During"] = dict()
        self.data["After"] = dict()

    def increment(self, action_duration):
        self.occurrences += 1
        self.duration += action_duration

    # storing a (count, duration) pair for each type of affect for each type of interval (within a given action)
    def increment_affect(self, interval, affect, duration):
        if affect not in self.data[interval].keys():
            self.data[interval][affect] = np.zeros(2)

        to_add = np.array([1, duration])
        self.data[interval][affect] += to_add

    def get_action_string(self, ordered_emotions):
        to_return = ""
        for e in ordered_emotions:
            for i in ["Before","During","After"]:
                try:
                    to_return += "%.4f,%.4f," % (self.data[i][e][0]/float(self.occurrences), self.data[i][e][1]/float(self.duration))
                except KeyError:
                    to_return += "0,0,"
        to_return += "%d," % self.occurrences
        return to_return


def create_header(actions, facets):
    to_return = "TestSubject,"
    for a in actions:
        for e in facets:
            for i in ["Before","During","After"]:
                for j in ["CountRate","DurationProp"]:
                    to_return += "%s-%s-%s-%s," % (i, a, e, j)
        to_return += "%s-Count," % (a)
    return to_return[:-1]


def get_emotions_from_header(header_list):
    emotions = []
    for h in header_list:
        h_split = h.split("-")
        if len(h_split) == 4:
            emotion = h_split[2]
            if emotion not in emotions:
                emotions.append(emotion)
    return emotions


def add_duration_column(df, summary_fp, keyword="All"):
    act_sum = pd.read_csv(summary_fp)
    duration_string_mod = "" if keyword == "All" else "-" + keyword
    act_sum.fillna(0, inplace=True)
    duration_df = act_sum.loc[:, ["TestSubject", "Duration%s" % duration_string_mod]]
    new_df = df.merge(duration_df, how='left', on="TestSubject")
    return new_df


# Add the grouped action types as columns for each type of window combination
def add_categorized_columns(windowed_file, output_filename, normalize=False, act_sum_filepath="C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/ActivitySummary/ActivitySummaryAppended.csv"):
    base_df = pd.read_csv(windowed_file)
    ordered_emotions = get_emotions_from_header(list(base_df.columns.values))
    if normalize:
        windowed_df = base_df.iloc[:, 1:].apply(lambda col: (col - col.mean()) / col.std(), axis=0)
        windowed_df.insert(loc=0, column="TestSubject", value=base_df["TestSubject"])
    else:
        windowed_df = base_df

    windowed_df.fillna(0, inplace=True)

    # Looping over naming convention to add relevant columns for a given action type (category)
    #  using globally defined action type dictionary
    for e in ordered_emotions:
        for i in ["Before", "During", "After"]:
            for j in ["CountRate", "DurationProp"]:
                for a_t in ACTION_TYPES.keys():
                    windowed_df["%s-%s-%s-%s" % (i, a_t, e, j)] = np.zeros(windowed_df.shape[0])
                    for a in ACTION_TYPES[a_t]:
                        windowed_df["%s-%s-%s-%s" % (i, a_t, e, j)] += windowed_df["%s-%s-%s-%s" % (i, a, e, j)]
                    windowed_df["%s-%s-%s-%s" % (i, a_t, e, j)] /= float(len(ACTION_TYPES[a_t]))

    # Getting just the action counts, which use a slightly different naming convention
    for a_t in ACTION_TYPES.keys():
        windowed_df["%s-Count" % a_t] = np.zeros(windowed_df.shape[0])
        for a in ACTION_TYPES[a_t]:
            windowed_df["%s-Count" % a_t] += windowed_df["%s-Count" % a]
        windowed_df["%s-Count" % a_t] /= float(len(ACTION_TYPES[a_t]))

    keyword = output_filename[output_filename.index("WindowedActions")+15:-14]
    windowed_df = add_duration_column(windowed_df, act_sum_filepath, keyword=keyword)
    windowed_df.to_csv(output_filename, index=False)


def create_windowed_action_file(event_facet_filename, output_filename, window_size, omit_AOIs=True, omit_AUs=True, show=False):
    start_time = datetime.now()

    with open(event_facet_filename, 'r') as ef_file, open(output_filename, 'w') as ofile:
        t_data = pd.read_csv(event_facet_filename, nrows=30000)
        ordered_actions = list(t_data["Event"].unique())
        ordered_emotions = [e for e in list(t_data["Target"].unique().astype(str)) if "Evidence" in e]
        ordered_actions.remove("FACET")
        if omit_AOIs:
            ordered_actions.remove("AOI")
        if omit_AUs:
            ordered_emotions = [e for e in ordered_emotions if "AU" not in e]
        output_header = create_header(ordered_actions, ordered_emotions)
        ofile.write(output_header+"\n")

        # Initialization
        header = ef_file.readline()
        ind = index_dictionary(header, [])
        action_list = data_list()
        facet_list = data_list()
        prev_test_subject = ""
        students = dict()  # dictionary from test ids to student data objects

        # Looping through all lines in the eventfacet file
        for line in ef_file:
            dl = data_line(line, ind, window_size)
            if (dl.line_type != "AOI" or not omit_AOIs) and dl.line_type != "NULL":
                current_test_subject = dl.test_subject
                if prev_test_subject != current_test_subject: # New test subject detected, need to output and reset
                    for a in action_list.list:  # Add all remaining actions to counts for student before writing
                        students[prev_test_subject].add_action_data(a, facet_list)
                    facet_list = data_list()  # Create new empty facet list
                    action_list = data_list()  # Create new empty action list
                    if prev_test_subject in students.keys():
                        students[prev_test_subject].write_student(ofile, ordered_actions, ordered_emotions)
                        if show:
                            print("%s wrote %s" % (datetime.now(), prev_test_subject))
                    students[current_test_subject] = student_data(dl) # Create new student data object

                actions_to_write = action_list.return_expired_actions(dl)
                for a in actions_to_write:
                    students[current_test_subject].add_action_data(a, facet_list)

                if dl.line_type != "FACET":
                    action_list.add_data(dl)
                else:
                    facet_list.add_data(dl)

                prev_test_subject = current_test_subject
                facet_list.trim(earliest_timestamp=action_list.return_earliest_before_ts())  # do not need to run this every iteration
        students[prev_test_subject].write_student(ofile, ordered_actions, ordered_emotions)

    minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60.
    print("Successfully created WindowedAction file: %s from: %s in %.3f minutes" % (output_filename, event_facet_filename, minutes_elapsed))


omit_AOIs = True  # if True only the 9 emotions are included, if false, all emotions + AUs included
omit_AUs = True
show = True
window_size = 5  # in seconds, for the before/after action window size
event_facet_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/EventSequenceFACETPostScan.csv"
output_filename = "C:/Users/robsc/Documents/NC State/GRAWork/CIData/Output/EventSequence/WindowedActionsPostScan.csv"

if __name__ == "__main__":
    create_windowed_action_file(event_facet_filename=event_facet_filename,
                                output_filename=output_filename,
                                window_size=window_size,
                                omit_AOIs=omit_AOIs,
                                omit_AUs=omit_AUs,
                                show=show)
    add_categorized_columns(windowed_file=output_filename,
                            output_filename=output_filename[:-4]+"Categories.csv",
                            normalize=True)