import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance

st.title("Dynamic Point Plotter and Distance Calculator")

# Function to calculate distances between consecutive points
def calculate_distances(points):
    distances = []
    for i in range(1, len(points)):
        dist = distance.euclidean(points[i-1], points[i])
        distances.append(f"Distance between point {i} and point {i+1}: {dist:.2f} pixels")
    return distances

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    st.write("Click on the image to add points. Points will be numbered sequentially.")

    # To store the points
    points = []

    # Capture user clicks
    x_clicks = st.number_input("X-coordinate")
    y_clicks = st.number_input("Y-coordinate")
    if st.button("Add Point"):
        points.append((int(x_clicks), int(y_clicks)))

    params = st.experimental_get_query_params()
    if 'points' in params:
        params_points = eval(params['points'][0])
        points = points + params_points
    st.experimental_set_query_params(points=str(points))

    if len(points) > 0:
        # Annotate points
        annotated_img = img.copy()
        for i, point in enumerate(points):
            cv2.circle(annotated_img, (point[0], point[1]), 5, (255, 0, 0), -1)
            cv2.putText(annotated_img, str(i + 1), (point[0] + 10, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        st.image(annotated_img, caption='Annotated Image with Points', use_column_width=True)

        # Calculate and display distances
        distances = calculate_distances(points)
        st.write("Distances between consecutive points:")
        for dist in distances:
            st.write(dist)