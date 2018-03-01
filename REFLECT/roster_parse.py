import pandas as pd

directory = "C:/Users/robsc/Documents/NC State/GRAWork/CI-REFLECT/Rosters/"
roster_numbers = ["1A", "1B", "2A", "2B", "3A", "3B"]
focus_students = ["Ruben Grundy",
                 "Bryan Mejia",
                 "Kaydin Dixon",
                 "Yeimi Perez-Santana",
                 "Georgia Nelson",
                 "Amirah McMahon",
                 "Charles Kellon",
                "Wisal Elhoudani"]
global_counter = 0


def pad_counter(counter):
    if counter < 10:
        string_counter = "000" + str(counter)
    elif counter < 100:
        string_counter = "00" + str(counter)
    else:
        string_counter = "0" + str(counter)
    return string_counter

full_data = pd.DataFrame()
for roster_number in roster_numbers:
    fp = "%sRoster %s.csv" % (directory, roster_number)
    roster_data = pd.read_csv(fp, skiprows=1)
    roster_data = roster_data.loc[pd.notnull(roster_data["Student Name"]), :]
    roster_data[["First Name", "Last Name", "Full Name"]] = roster_data["Student Name"].apply(
        lambda cell: pd.Series({"First Name": cell.split(",")[1].strip(),
                                "Last Name": cell.split(",")[0].strip(),
                                "Full Name":cell.split(",")[1].strip() + " " + cell.split(",")[0].strip()}, index=["First Name", "Last Name", "Full Name"]))
    roster_data["Password"] = roster_data["Student Number"]
    roster_gens = []
    for _ in range(roster_data.shape[0]):
        global_counter += 1
        roster_string = "%s%s" % (roster_number, pad_counter(global_counter))
        roster_gens.append(roster_string)
    roster_data["Login"] = pd.Series(roster_gens)
    roster_data["Focus Participant"] = roster_data["Full Name"].apply(lambda cell: int(cell in focus_students))
    roster_data.to_csv("%sRoster %s Revised.csv" % (directory, roster_number), index=False)
    full_data = pd.concat([full_data, roster_data])
print(full_data.shape)
full_data.to_csv(directory+"Full Roster.csv", index=False)
