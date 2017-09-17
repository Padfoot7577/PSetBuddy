from flask import Flask
from flask import render_template
from flask import request
import train

website = Flask(__name__)

@ website.route('/')
def index():
	return render_template("index.html")

def update_index_db(info_vector):
	with open("student_info.csv", 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', 
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(info_vector)
    return sum(1 for line in open("student_info.csv")) - 2

def update_vector_db(raw_vector):
	vector = raw_vector[0:len(raw_vector) - 1]
	student_classes = raw_vector[-1].replace(" ", "").upper().split(";")
	# Remove repetitions and NONE's:
	student_classes = set(student_classes)
	student_classes.discard("NONE")
	student_classes = list(student_classes)
	total_classes_list = []
	additional_col = 0
	with open(csvFileName, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(filewiter): # each row is 1 student
            if i == 0:
            	total_classes_list = row[7:]
            	for course in student_classes:
            		if course not in total_classes_list:
            			row.append(course)
            			additional_col += 1
            	total_classes_list = row[7:]
            else:
            	row.append([0] * additional_col)
            filewriter.writerow(row)
        class_vector = [0]*len(total_classes_list)
        for course in student_classes:
            ind = total_classes_list.index(course)
            class_vector[ind] = 1
        vector.extend(class_vector)
        filewriter.writerow(vector)
    # with open("student_data_for_import.csv", 'a') as csvfile:
    #     filewriter = csv.writer(csvfile, delimiter=',', 
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     filewriter.writerow(vector)

@website.route('/',methods=["POST"])
def dataasp():
	fn=request.form["firstname"]
	ln=request.form["lastname"]
	email=request.form["email"]

	dic={"info":[email, fn, ln], "vectors":[]}

	dic["vectors"].append(request.form["classyear"])
	dic["vectors"].append(request.form["major1"])
	dic["vectors"].append(request.form["major2"])
	dic["vectors"].append(request.form["studyplace"])
	dic["vectors"].append(request.form["time"])
	dic["vectors"].append(request.form["animal"])
	dic["vectors"].append(request.form["residence"])
	dic["vectors"].append(request.form["classes"])
	
	database_index = update_index_db(dic["info"])
	update_vector_db(dic["vectors"])
	train.clusterize()
	buddy_id = find_buddy(database_index) + 1

	with open("student_info.csv", newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(datareader):
        	if i == buddy_id:
        		email = row[0]
				fname = row[1]
				lname = row[2]

	graphics=render_template("getmatched.html", f_name=fname, l_name=lname, email=email)

	return graphics 




if __name__ == '__main__':
	website.run(debug=True)