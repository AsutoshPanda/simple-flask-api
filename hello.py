from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return '{  
    "employee": {  
        "name":       "Asutosh",   
        "salary":      -500,   
        "married":    "to-be"  
    }  
}'

if __name__ == '__main__':
    app.run()
