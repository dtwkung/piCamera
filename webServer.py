from flask import Flask, render_template, Response
from flask_basicauth import BasicAuth
from moveDetect import moveDetect

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'dtwkung'
app.config['BASIC_AUTH_PASSWORD'] = 'testtest'
app.config['BASIC_AUTH_FORCE'] = True
basic_auth = BasicAuth(app)

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')


def gen(detector):
    while True:
        frame = detector.getFrame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(moveDetect(model='mobilenet')),
                    mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True, ssl_context='adhoc', port=53189)
