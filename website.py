from flask import Flask
from flask import render_template
from flask import request

website = Flask(__name__)

@ website.route('/')
def index():
	return render_template("index.html")

@website.route('/',methods=["POST"])
def dataasp():
	fn=request.form["firstname"]
	ln=request.form["lastname"]
	email=request.form["email"]

	dic={"info":[fn+' '+ln,email], "vectors":[]}

	dic["vectors"].append(request.form["classyear"])
	dic["vectors"].append(request.form["major1"])
	dic["vectors"].append(request.form["major2"])
	dic["vectors"].append(request.form["studyplace"])
	dic["vectors"].append(request.form["time"])
	dic["vectors"].append(request.form["animal"])
	dic["vectors"].append(request.form["residence"])
	dic["vectors"].append(request.form["classes"])
	print (dic)

	return email




if __name__ == '__main__':
	website.run(debug=True)