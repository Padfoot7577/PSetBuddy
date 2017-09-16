from flask import Flask, request, render_template
app = Flask(__name__)

@app.route("/")
def hello():
	return """
			<h1>Name Reverser</h1> 
			<input type="text"></input> 
			<button>OK!</button>
		"""

@app.route("/yo")
def random_stuff():
	return "<b>Yo!<b>"	

@app.route("/reverse")
def rea():
	name = request.args.get("name")
	return name[::-1]

if __name__ == "__main__":
	app.run(debug=True)