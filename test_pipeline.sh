gst-launch-1.0 filesrc location=/data/in/test.png ! decodebin ! videoconvert ! mozza deform=/data/out/test.dfm alpha=1 ! videoconvert ! jpegenc ! filesink location=/data/out/transformed.jpg
