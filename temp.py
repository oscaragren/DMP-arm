from hardware.can.can_message_parser import CANMessageParser
from hardware.can.can_socketcan import SocketCANInterface
import time

can_message_parser = CANMessageParser()
can_interface = SocketCANInterface(interface="can0", bitrate=1000000)
can_interface.start()

time.sleep(1)
movement_command_rate_ms = 500

# joints order: elbow, shoulder up down, shoulder left right, shoulder twist.
joint_angles_sequence = [[0,0,0,0],[0,0,0,0]] # Example

def index_to_name(index):
    if index == 0:
        return "robot_elbow_up_down_actuation"
    elif index == 1:
        return "robot_shoulder_up_down_actuation"
    elif index == 2:
        return "robot_shoulder_left_right_actuation"
    elif index == 3:
        return "robot_upper_arm_rotation_actuation"
    else:
        raise "Error converting index to name"

def angles_to_messages(angles):
    return [can_message_parser.encode(index_to_name(i), {"value": angle}) for i, angle in enumerate(angles)]

can_messages_to_send_sequence = [
    angles_to_messages(angles) for angles in joint_angles_sequence
]

first_move = True
prev_time_ns = time.monotonic_ns()
for msgs in can_messages_to_send_sequence:
    for can_id, data in msgs:
        print(f"sending id {hex(can_id)}, data {data}")
        success = can_interface.send(can_id, data)
        if success:
            print(f"Sent {can_id} with data {data}, tx_count: {can_interface.tx_count}")
        else:
            print(f"Error sending {can_id} with data {data} over can!")

    if first_move:
        first_move = False
        time.sleep(10)
        prev_time_ns = time.monotonic_ns()
    time_ns = time.monotonic_ns()
    expected_new_time = prev_time_ns + movement_command_rate_ms * 1_000_000
    time_to_sleep_ns = expected_new_time - time_ns
    if time_to_sleep_ns > 0:
        time.sleep(time_to_sleep_ns / 1_000_000_000)
    prev_time_ns = expected_new_time

time.sleep(1)
can_interface.stop()