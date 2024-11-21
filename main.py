import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化深度相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# 加载MobileNet SSD模型
model_path = 'deploy.prototxt'
weights_path = 'mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

# 设置深度图像的伪彩色映射
colorizer = rs.colorizer()

try:
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 将RGB帧数据转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 应用伪彩色映射到深度帧并转换为numpy数组
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # 在RGB图像上进行人体检测
        (h, w) = color_image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        closest_depth_value = float('inf')  # 初始设为正无穷大

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 置信度阈值
                class_id = int(detections[0, 0, i, 1])
                if class_id == 15:  # 15是人体的类别ID
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # 获取人体框中心点附近的深度数据
                    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    depth_data = np.asanyarray(depth_frame.get_data())

                    # 限制深度数据的有效区域
                    depth_roi = depth_data[startY:endY, startX:endX]
                    depth_roi = depth_roi * depth_scale

                    # 排除无效深度值并计算深度值
                    valid_depth_roi = depth_roi[np.nonzero(depth_roi)]
                    if len(valid_depth_roi) > 0:
                        depth_value = np.mean(valid_depth_roi)
                    else:
                        depth_value = float('inf')  # 没有有效深度时设为无穷大

                    # 更新最近深度值
                    closest_depth_value = min(closest_depth_value, depth_value)

                    # 显示距离信息
                    cv2.rectangle(color_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # 显示最近人物框中的距离信息
        if closest_depth_value != float('inf'):
            cv2.putText(color_image, f'Closest Distance: {closest_depth_value:.2f} m', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # 显示RGB图像和伪彩色深度图像
        cv2.imshow('RGB Image', color_image)
        cv2.imshow('Colorized Depth Image', colorized_depth)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
