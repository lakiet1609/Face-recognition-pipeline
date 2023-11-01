# DEPLOY FACE RECOGNITION SYSTEM

## FACE RECOGNITION 

There are 3 main processes inside Face System including:
- SCRFD process: Get the face detection (bounding boxes, keypoints ...)
- ARCFACE process: Get face vectors (embeddings)
- FAISS process: Calculate the inner product of the 2 different vectors to recognize face

![4235](https://github.com/lakiet1609/face_project1/assets/116550803/48668198-0db6-4bec-83f3-848172a0f36d)

### SCRFD (Using Onnxruntime Inference)

#### Applying SCRFD to face detected 

- Input: image and desired image size
- Output: predicted bounding boxes (5,) and keypoints 


### ARCFACE (Using Onnxruntime Inference)

#### Using ARCFACE to get face embeddings (vectors)
- Input: image and keypoints
- Ouput: face embeddings with dimension (512,)


### FAISS

#### Faiss implementation to calculating the difference of face embeddings
- Input: embedding vectors and nearest neighbors
- Output: person_id, score ...


## PROCESSES OF FACE RECOGNITION SYSTEM

SCRFD PROCESS -> ARCFACE PROCESS -> FAISS PROCESS

1. STEP 1: 

- Initiate SCRFD process to get face detection

![Screenshot from 2023-11-01 22-44-24](https://github.com/lakiet1609/face_project1/assets/116550803/c9bc4cf4-4dc3-4f0f-bcbb-26dd793b1b49)

2. STEP 2:

- Applying the results of SCRFD process (keypoints) -> implement ARCFACE process to gain vectors

![Screenshot from 2023-11-01 23-08-08](https://github.com/lakiet1609/face_project1/assets/116550803/70ff6f52-9cc8-48fd-9ae2-3dfdb30cb6aa)

3. STEP 3:

- Push the predicted face embeddings values onto MONGO DATABASE to store the value of each face


4. STEP 4: 

- Applying FAISS method to differentiate between face embeddings by calculating the inner product


5. STEP 5:

- Deploy it using FASTAPI 

![Screenshot from 2023-11-01 23-12-54](https://github.com/lakiet1609/face_project1/assets/116550803/11d3ac08-f982-4c0c-91d3-61848e3e50ff)


