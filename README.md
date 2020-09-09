# THUCourseSpider
It is a Python script to automatically login to the Tsinghua University 
courses selection system and keeps trying to select the courses brutely.

- `label-captcha.py`: Label the captcha with their corresponding code manually and verify its correctness through Tsinghua course system
- `segment-captcha.py`: Segment the captcha into characters using a GUI manually
- `prepare-dataset.py`: Prepare and split dataset
- `classifier.py`: A simple CNN to classify the segmented characters
