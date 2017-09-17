import csv

def clean_raw_data(csvFileName):
	survey_data = []
	with open(csvFileName, newline='') as csvfile:
		datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for i, row in enumerate(datareader): # each row is 1 student
			if i > 0:
				student = [parse_year(row[0]), parse_major(row[1]), parse_major(row[2]),
					   	   parse_location(row[3]), parse_time(row[4]), parse_pet(row[5]),
					   	   parse_residence(row[6]), row[find_year(row)].replace(" ", "").upper().split(";")]
				survey_data.append(student)
	#print(survey_data)
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
		  "" : 0,
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

if __name__ == "__main__":
	clean_raw_data("SurveyDataProcessed1.csv")