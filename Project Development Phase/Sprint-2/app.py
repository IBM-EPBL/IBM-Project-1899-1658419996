from flask import Flask, render_template, Response, url_for
import cv2

app = Flask(__name__)

def gen():
    vid = cv2.VideoCapture(0)
    while (True):
        ret, frame = vid.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, image_binary = cv2.imencode('.jpg',gray_image)
        _, col_img = cv2.imencode('.jpg',frame)
        binary_data = image_binary.tobytes()
        col_binary = col_img.tobytes()
        data_frame = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + binary_data + b'\r\n\r\n'
        col_data_frame = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + col_binary + b'\r\n\r\n'
        yield col_data_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)
