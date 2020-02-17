import numpy as np
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


def rearange(line):
    x = []
    y = []
    for p in line:
        x.append(p[0])
        y.append(p[1])
    return x,y

def convex_hull(pts):

    conv_H = ConvexHull(np.array(pts))
    if conv_H is None:
            return None
    x = []
    y = []
    j = 0
    for i in conv_H.vertices:
        x_,y_ = pts[i]
        if j < 1:
            x.append(x_)
            y.append(y_)
            j += 1
            continue
        j = len(x)
        
        p0,p1 = rearange(get_line((x[j-1],y[j-1]),(x_,y_)))

        x.extend(p0)
        y.extend(p1)
        x.append(x_)
        y.append(y_)
                        


    p0,p1 = rearange(get_line((x[-1],y[-1]),(x[0],y[0])))
    x.extend(p0)
    y.extend(p1)
    h_i = (np.array(x),np.array([y]))
        
    
    return h_i

def sequentialLabeling(img,max_dist=1,threshold=2):

    """
    
        Label clouds and return array of tuble (label,cloudsize)
        labels are sorted in descending order by cloudsize
        Explicit location of cloud labeld by label_A 
        can be found by np.where(img == label_A)


    """
    img = img.copy().astype(np.uint32)
    img[img >= threshold] = 1

    true_value = img.max()
    x,y = np.where(img == 1)


    collision = dict()
    label = 2
    
    for i,j in zip(x,y):
        i_X = slice(i-max_dist,i+max_dist)
        j_Y = slice(j-max_dist,j+max_dist)

        window = img[i_X,j_Y]

        neighbours = np.argwhere(window > 1)


        if len(neighbours) == 0:
            window[window == 1] = label
            label +=1
            img[i_X,j_Y] = window

        elif len(neighbours) == 1:
            window[window == true_value] = window[neighbours[0,0],neighbours[0,1]]
            img[i_X,j_Y] = window


        # handle label collisions

        else:
            k = np.amax(window)
            img[i,j] = k
            for index in neighbours:
                nj = window[index[0], index[1]]

                if nj != k:
                    if k not in collision:
                        collision[k] = set()
                    collision[k].add(nj)
                    if collision[k] is None:
                        del collision[k]


    def changeLabel(elem):
        c_label = collision[elem]
        for l in c_label:
            img[img == l] = elem


    def rearangeCollisions():
        for elem in collision:
            for item in collision[elem]:
                if item in collision:
                    collision[elem] = (collision[elem] | collision[item])
                    collision[item] = set()
            if elem in collision[elem]:
                collision[elem].remove(elem)


    rearangeCollisions()


    for i,elem in enumerate(collision):
        if collision[elem] is None:
            continue
        changeLabel(elem)

    cloud_size = []

    for i in range(2,label):
        idx = np.where(img == i)
        a = len(idx[0])

        if a == 0:
            continue
        cloud_size.append((i,a,idx))
    cloud_size = sorted(cloud_size, key=lambda x: x[1],reverse = True)

    return cloud_size