import numpy as np
from errors import WrongConstant
from heapq import *
from point import Point

class K_NN:
    """Initialisation of the k nearest neighbours model

    Attributes:
        train_set_x -- the training set of the X input
        train_set_y --  the training set of the Y input
        learning_rate -- the learning rate of the model, must be greater than 0
    """
    def __init__(self, train_set_x, train_set_y, k, metric = "infinite"):
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.k = k
        self.metric = metric
        self.associate_points_to_class()
    """Training the model"""

    def train(self):
        pass

    """Associates each point to its class in a dictionnary"""

    def associate_points_to_class(self):
        self.association = {}

        for i in range(len(self.train_set_x)):
            self.association[tuple(self.train_set_x[i])] = self.train_set_y[i]


    """Returns the closest k points to a given point 

    Attributes:
        point -- point for which the closests k points are compute
    """
    def closest_k_points(self, point):
        point = Point(point)
        k_closest_points = []
        for index in range(len(self.train_set_x)):
            sample_point =  Point(self.train_set_x[index])
            if index < self.k:
                heappush(k_closest_points, (-sample_point.distance_from_point(point), sample_point, index))
            else:
                heappop(k_closest_points)
                heappush(k_closest_points, (-sample_point.distance_from_point(point), sample_point, index))
        
        array_of_k_closest_points = []

        for i in range(len(k_closest_points)):
            (_, point, index) = heappop(k_closest_points)
            array_of_k_closest_points.append(point.array_form())

        return array_of_k_closest_points

    """Predicts the classification based on the input

    Attributes:
        point_to_classify -- the x input to classify
    """

    def predict(self, x_to_classify):
        k_closest_points = self.closest_k_points(x_to_classify) 

        classes_count = {}
        max_count = 0
        print(k_closest_points)
        for x in k_closest_points:
            tuple_form_of_x =  tuple(x)
            if self.association[tuple_form_of_x] not in classes_count:
                classes_count[self.association[tuple_form_of_x]] = 0
            classes_count[self.association[tuple_form_of_x]] += 1

            if classes_count[self.association[tuple_form_of_x]] > max_count:
                max_count = classes_count[self.association[tuple_form_of_x]]
                predicted_class =  self.association[tuple_form_of_x]
        return predicted_class

