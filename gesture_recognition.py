import math

class gesture:
    def __init__(self, landmark):
        self.landmark = landmark

    def finger_check(self):
    
        if len(self.landmark) < 21:
            return 'bad detection'
        
        #wriet = self.landmark[0]
        thumb = self.landmark[1:5]
        index = self.landmark[5:9]
        middle = self.landmark[9:13]
        ring = self.landmark[13:17]
        pinky = self.landmark[17:]
    
        str_fingers = [self.straight_finger(thumb), self.straight_finger(index), self.straight_finger(middle), self.straight_finger(ring), self.straight_finger(pinky)]
    
        if self.is_straight_finger(str_fingers, [1]):
            if self.finger_up(index):
                return "up"
    
        return ""

    def hand_center(self):
        x = self.landmark[0].x + self.landmark[1].x + self.landmark[5].x + self.landmark[9].x + self.landmark[13].x + self.landmark[17].x
        y = self.landmark[0].y + self.landmark[1].y + self.landmark[5].y + self.landmark[9].y + self.landmark[13].y + self.landmark[17].y
        return [x/6, y/6]
    
    
    def straight_finger(self, finger) :
        error_angle = 10
    
        mcp, pip, dip, tip = finger[0], finger[1], finger[2], finger[3]
    
        k1x, k1y = (mcp.x - pip.x)/self.finger_point_distance(mcp, pip), (mcp.y - pip.y)/self.finger_point_distance(mcp, pip)
        k2x, k2y = (pip.x - dip.x)/self.finger_point_distance(pip, dip), (pip.y - dip.y)/self.finger_point_distance(pip, dip)
        k3x, k3y = (dip.x - tip.x)/self.finger_point_distance(dip, tip), (dip.y - tip.y)/self.finger_point_distance(dip, tip)
        
        if abs(k1x*k2y - k2x*k1y) < math.sin(math.pi*error_angle/180) and abs(k3x*k2y - k2x*k3y) < math.sin(math.pi*error_angle/180):
            return True
        else:
            return False
    
    def is_straight_finger(self, fingers, require):
        result = 0
        all = [0,1,2,3,4]
        unrequire = list(set(all) - set(require))
        for index in range(len(require)):
            if fingers[require[index]]:
                result += 1
        for index in range(len(unrequire)):
            if fingers[unrequire[index]]:
                result -= 1
        return result == len(require)
    
    
    def finger_up(self, finger):
        error_angle = 10
    
        mcp,tip = finger[0], finger[3]
        ky = (tip.y - mcp.y)/self.finger_point_distance(mcp, tip)
        if -ky > math.sin(math.pi*(1/2 - error_angle/180)):
            return True
        return False
    
    def finger_point_distance(self, point1, point2):
        return math.sqrt(math.pow(point1.x - point2.x, 2) + math.pow(point1.y - point2.y, 2))

    def depth_to_camera(self):
        dep = self.landmark[0].z
        for point in self.landmark:
            if dep > point.z:
                dep = point.z
        return dep



if __name__ == '__main__':
    fingers = [True,True,True,True,True]
    require = [1,3,4]
    result = 0
    all = [0,1,2,3,4]
    unrequire = list(set(all) - set(require))
    print(unrequire)
    for index in range(len(require)):
        if fingers[require[index]]:
            result += 1
    for index in range(len(unrequire)):
        if fingers[unrequire[index]]:
            result -= 1
            
    print(result == len(require))