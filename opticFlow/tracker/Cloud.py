
import numpy as np
import cv2 as cv
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def get_line(start, end):

    """
        
        Stolen from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm

    """
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

class Cloud():
    def __init__(self,points,size = None):
        
        self.points = points
        if size is None:
            self.size = len(points[0])
        else:
            self.size = size
        self.color = [255,255,255]
        self.center_of_mass = int(np.sum(points[0])/len(points[0])),int(np.sum(points[1])/len(points[1]))
        self.min_x,self.max_x = points[0].min(),points[0].max()
        self.min_y,self.max_y = points[1].min(),points[1].max()  
        self.direction_vec = None

        # self.pts == self.points as array not index_array
        self.pts = [[x,y] for x,y in zip(self.points[0],self.points[1])]
        if len(self.pts) < 10:
            self.hull = None
        else:
            self.hull = ConvexHull(np.array(self.pts))



    def __lt__(self, other):
        return self.size < other.size

    def __eq__(self, other):
        return self.size == other.size
    
    def __str__(self):
        return "Cloudsize: {}\nCenter : {}\n".format(self.size,self.center_of_mass)
    
    def addText(self,img,coord,text):

        font                   = cv.FONT_HERSHEY_COMPLEX_SMALL
        fontScale              = 0.5
        fontColor              = (255,255,255)
        lineType               = 1

        cv.putText(img,
                   text, 
                   coord,
                   font, 
                   fontScale,
                   fontColor,
                   lineType)


        return img

    def draw_hull(self,img):


        def rearange(line):
            x = []
            y = []
            for p in line:
                x.append(p[0])
                y.append(p[1])
            return x,y

        if self.hull is None:
            return img
        #pts = ( np.array(self.hull[0]),np.array(self.hull[1]))

        x = []
        y = []
        j = 0
        for i in self.hull.vertices:
            x_,y_ = self.pts[i]
            if j < 1:
                x.append(x_)
                y.append(y_)
                j += 1
                continue
            j = len(x)
            print((x[j-1],y[j-1]),(x_,y_))
            p0,p1 = rearange(get_line((x[j-1],y[j-1]),(x_,y_)))

            x.extend(p0)
            y.extend(p1)
            x.append(x_)
            y.append(y_)
            
            print()
            


        #print(self.hull)
        p0,p1 = rearange(get_line((x[-1],y[-1]),(x[0],y[0])))
        x.extend(p0)
        y.extend(p1)
        h_i = (np.array(x),np.array([y]))
        

        self.convex_hull_pts = h_i
        if len(img.shape) == 3:
            img[h_i] = [255,255,255]
            cv.circle(img, 
                (self.center_of_mass[1],self.center_of_mass[0]), 
                5, 
                [0,0,255], 
                thickness=1, 
                lineType=8, 
                shift=0) 
        
        else:
            img[h_i] = 255

        return img


    def bounding_box(self,img):


        if self.size <= 1:
            return img
  
        ret = img
        ret[self.min_x:self.max_x,self.min_y] = 255
        ret[self.min_x:self.max_x,self.max_y] = 255
        ret[self.min_x,self.min_y:self.max_y] = 255
        ret[self.max_x,self.min_y:self.max_y] = 255
            


        ret = self.addText(ret,
                        (self.max_y,self.max_x+50),
                        "SWP: ("+str(self.center_of_mass[0])+","+str(self.center_of_mass[1])+")")
            
        ret[self.max_x:self.max_x+40,self.max_y] = 255

        return ret
    
    
    def bbox(self,img):
        return self.bounding_box(img)
    
    def paintcolor(self,img):
        img[self.points] = self.color
        return img
        
    def points_to_2D(self):
        ret = np.array(self.points)
        x,y = ret.shape
        return ret.reshape(y,1,x)

    def set_direction(self,dirv):
        if self.direction_vec is None:
            self.direction_vec = dirv
        else:
            # updaten
            pass


    def is_inPath(self,point,threshold = 5):
        
        """
            point = pt + t * direction
        """
        if self.direction_vec is None:
            return False

        point = np.array(list(point))
        t = []
        for x,y in zip(self.points[0],self.points[1]):
            pts = np.array([x,y])
            """
            print(pts)
            print(point)
            print(self.direction_vec)
            """
            t = (point - pts)  / self.direction_vec
            #if int(t[0]) == int(t[1]):
            if t[0]+threshold > t[1] and t[0]-threshold < t[1] or t[1]+threshold > t[0] and t[1]-threshold < t[0]:
                print("WAS")
                print(t)
                print("----")
                return True
        return False




    def dist(self,cloud):
        """
        
            returns euclidean dist between center of ma
        
        """
        
        return np.sqrt((cloud.center_of_mass[0] - self.center_of_mass[0])**2 +
                       (cloud.center_of_mass[1] - self.center_of_mass[1])**2)