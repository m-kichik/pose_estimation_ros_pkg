Этот репозиторий содержит package для pose estimation с использованием YOLOv7

```
source /opt/ros/<your distro name>/setup.bash
```

### Подготовка рабочего пространства
```
mkdir ~/catkin_ws/src -p
```

### Клонирование этого репозитория
```
cd ~/catkin_ws/src/
git clone https://github.com/m-kichik/pose_estimation_ros_pkg .
```

###
```
cd ~/catkin_ws/src/yolov7_pkg/scripts/
chmod +x image_publisher.py
chmod +x yolov7_node.py
```

### Создание папки для весов модели
```
cd ~/catkin_ws/src/yolov7_pkg/
mkdir src/weights -p
```
В папку weights необходимо положить файл [yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

### Создание рабочего пространства
```
cd ~/catkin_ws/
catkin_make
```

Теперь в файлах image_publisher.py и yolov7_node.py необходимо отредактировать первую строку: нужно добавить путь к вашему интерпретатору, куда необходимо также поставить библиотеки из yolov7/requirements.txt (pip install -r /<path to file>/requirements.txt)

## Запуск узлов
Первый терминал:
```
source /opt/ros/<your distro name>/setup.bash
roscore
```

Второй терминал:
```
cd ~/catkin_ws/
source devel/setup.bash
rosrun yolov7_pkg yolov7_node.py
```

Третий терминал:
```
cd ~/catkin_ws/
source devel/setup.bash
rosrun yolov7_pkg image_publisher.py <path to your image>
```
