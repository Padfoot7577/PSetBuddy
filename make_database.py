import csv

"""
This method takes a 2*2 matrix (nested lists).
the last column of the list represent all the classes each person takes
step 1: loop through each person's classes, add it in the set
step 2: make the set into a list, sort the list
step 3: check each person's classes against the master list, create vector
with 1s being yes have taken, 0s being no, have not taken
the output should be a list of lists, the lists nested within are each persons's individual class vector 
"""
def generate_class_vectors(input_l):
    total_classes=set()
    for index in range(1, len(input_l)):
        for each_class in input_l[index][-1]:
            total_classes.add(each_class)
    # print(total_classes)
    total_classes_list=list(total_classes)
    total_classes_list.sort()
    # TODO: total_classes_list = find_total_classes(input_l)
    # Use to replace all lines above
    labels = input_l[0]
    labels.extend(total_classes_list)
    # print(labels)
    # print(total_classes_list)
    everyone_classes=[labels]
    for i in range(1, len(input_l)):
        person = input_l[i]
        classes=person[-1]
        class_vector=[0]*len(total_classes_list)
        for c in classes:
            ind=total_classes_list.index(c)
            class_vector[ind]=1
        personal_data = person[0:len(person)-1]
        personal_data.extend(class_vector)
        everyone_classes.append(personal_data)
    return everyone_classes 

def clean_raw_data(csvFileName):
    survey_data = []
    with open(csvFileName, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(datareader): # each row is 1 student
            if i == 0:
                survey_data.append(row[0:7])
            else:
                student = [parse_year(row[0]), parse_major(row[1]), parse_major(row[2]),
                           parse_location(row[3]), parse_time(row[4]), parse_pet(row[5]),
                           parse_residence(row[6]), row[find_year(row)].replace(" ", "").upper().split(";")]
                survey_data.append(student)
    return survey_data

def find_year(row):
    for i in range(len(row)-1, 6, -1):
        if row[i].upper() != "NONE":
            return i
    raise Exception("Missing course input!")

def parse_year(str):
    year_switch = {
        "2021" : 0,
        "2020" : 1,
        "2019" : 2,
        "2018" : 3,
        "2017" : 4,
           "G" : 4
    }
    return year_switch[str]

def parse_major(str):
    major_switch = {
      "None" : 0,
         "1" : 1,
         "2" : 2,
         "3" : 3,
         "4" : 4,
         "5" : 5,
         "6" : 6,
         "7" : 7,
         "8" : 8,
         "9" : 9,
        "10" : 10,
        "11" : 11,
        "12" : 12,
        "14" : 13,
        "15" : 14,
        "16" : 15,
        "17" : 16,
        "18" : 17,
        "20" : 18,
        "21" : 19,
        "22" : 20,
        "24" : 21
    }
    return major_switch[str]

def parse_location(str):
    loc_switch = {
        "Library" : 0,
        "Dorm" : 1,
        "Commons/lounge" : 2,
        "Cafe" : 3,
        "None of the above" : 4
    }
    return loc_switch[str]

def parse_time(str):
    time_switch = {
        "Morning" : 0,
        "Night" : 1,
        "24/7" : 2
    }
    return time_switch[str]

def parse_pet(str):
    pet_switch = {
        "Dogs" : 0,
        "Cats" : 1,
        "Both" : 2,
        "Neither" : 3
    }
    return pet_switch[str]

def parse_residence(str):
    residence_switch = {
        "Cambridge" : 0,
        "Boston" : 1,
        "Other" : 2,
    }
    return residence_switch[str]

def csv_writer(csvfileName, data):
    """ Write a CSV document containing data.
    """
    with open(csvfileName, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', 
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            filewriter.writerow(row)
    print("Write complete.")

if __name__ == "__main__":

    #test case 1 represents four people and last element representing their current classes
    test_case_1=[[[2],[0],['5.111','7.012','CC201']],
             [[15],[1],['7.012','15.401','6.006','6.036']],
             [[3],[1],["3.091","2.001","21M.900"]],
             [[6],[0],["6.01","6.031","6.006","1.02"]]]
    # print(generate_class_vectors(test_case_1))

    data = clean_raw_data("SurveyDataProcessed1.csv")
    final_data = generate_class_vectors(data)
    #print(data)
    #print(generate_class_vectors(data))
    # for row in final_data:
    #     print(len(row))
    print(final_data)
    # csv_writer("student_data_for_import.csv", final_data)
    
