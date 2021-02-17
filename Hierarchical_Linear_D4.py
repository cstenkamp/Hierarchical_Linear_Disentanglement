# libraries
##############################################################################
##############################################################################
import gc
import itertools
import math
import os
import re
import time
import warnings

import numpy as np

# from sklearn.utils import class_weight
import scipy as sp
from sklearn import linear_model
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)

warnings.filterwarnings("ignore")
print("finish importing")


# In[ ]:


class NotTrainedError(Exception):
    pass


class NotFitToCorpusError(Exception):
    pass


class Primary_Sub_Features:
    # ent,fea,orderd_features,mdsMAIN,pmiweightsOriginal,orderd_features_directions,orderd_features_Kappa,facets_Path)
    def __init__(
        self,
        ent,
        fea,
        orderd_features,
        mds,
        train_labels,
        orderd_features_directions,
        orderd_features_Kappa,
        primary_sub_Path,
    ):

        self.__fea = fea  # features extracted from the corpus
        self.__ent = ent  # documents names
        self.__orderd_features = (
            orderd_features  # features ordered based on Kappa score
        )
        self.__orderd_features_directions = (
            orderd_features_directions  # directions for the self.__orderd_features
        )
        self.__orderd_features_Kappa = (
            orderd_features_Kappa  # Kappa score for each feature
        )

        self.__mds = mds
        self.__train_labels = train_labels
        self.__primary_sub_Path = primary_sub_Path

    def findingPrimaryFeatures(self):

        """
        This finction will return the Primary features .
        """
        (
            self.__primarfeatures,
            self.__primaryFDirections,
            self.__primaryFPosIns,
            self.__prim_PositveIns_vectors,
            self.__prim_PositveIns_Labels,
            self.__Prim_feature_Index,
            self.__clusterprediction,
        ) = self.__clusAff(
            self.__orderd_features_directions[:5000], self.__orderd_features[:5000]
        )
        return self.__primaryFDirections

    def LearningFeaturesDirecInPr(self):

        """
        This finction will learn the directions for each term in the Primary features using the primary features' region .
        """

        sub_feat = []
        sub_feat_directionsfiltered = []
        allKappaTest = []
        allorderdFe = []
        for pf in range(len(self.__primarfeatures)):

            directionsLGR_ = []
            # directionsLGR_test = []
            kappa_Score_all_ = []
            kappa_Score_all_test_ = []
            # accuracy = []
            # pmiTotal = []

            if len(self.__Prim_feature_Index[pf]) != 0:
                # start_time = time.time()

                for fe in self.__Prim_feature_Index[pf]:

                    (
                        kappa_scoretest,
                        kappa_score1,
                        f1,
                        direction,
                        acc,
                        TP,
                        FP,
                        TN,
                        FN,
                        predicted,
                        probabilities,
                    ) = runLR1(
                        self.__prim_PositveIns_vectors[pf],
                        np.array(self.__prim_PositveIns_Labels[pf])[:, fe],
                    )
                    directionsLGR_.append(direction)
                    kappa_Score_all_.append(kappa_score1)
                    kappa_Score_all_test_.append(kappa_scoretest)
                orderd_accuracy_index_ = np.argsort(kappa_Score_all_test_)[::-1]
                # orderd_accuracy_index_all = np.argsort(kappa_Score_all_)[::-1]
                orderd_features_ = []
                orderd_features_directions_ = []

                for js in orderd_accuracy_index_:
                    orderd_features_.append(
                        self.__fea[self.__Prim_feature_Index[pf][int(js)]]
                    )
                    orderd_features_directions_.append(directionsLGR_[int(js)])

                tempsubFe = []
                directionsLGR_TempFiltered = []

                for f in range(0, len(orderd_features_)):
                    #
                    # if orderd_features_[f] not in orderd_features[:100]:
                    if kappa_Score_all_test_[orderd_accuracy_index_[f]] >= 0.1:
                        print(
                            orderd_features_[f],
                            kappa_Score_all_test_[orderd_accuracy_index_[f]],
                            "****",
                        )

                        tempsubFe.append(orderd_features_[f])
                        directionsLGR_TempFiltered.append(
                            orderd_features_directions_[f]
                        )

            sub_feat.append(tempsubFe)
            allKappaTest.append(kappa_Score_all_test_)
            allorderdFe.append(orderd_features_)
            sub_feat_directionsfiltered.append(directionsLGR_TempFiltered)
        print(
            "step11 learning the rest of the featuer directions in  the primary features part of the space is finished"
        )

        # clustering the rest of the featuer directions  To obtain the sub-features (the second level)
        ##############################################################################
        ##############################################################################

        self.__cluster2stLevel_index_all = []
        self.__clusterSubFeatuer_all = []
        self.__clusterSubFeatuer_all_kappa = []
        for i in range(0, len(sub_feat)):
            print("______________________________________________________")
            print("primary feature number: ", i)
            print(self.__primarfeatures[i])
            print()
            print()
            cluster2stLevel_index = []
            clusterSubFeatuer = []
            clusterSubFeatuer_kappa = []

            print()
            if len(sub_feat[i]) >= 1:
                clus = AffinityPropagation().fit(
                    np.array(sub_feat_directionsfiltered[i])
                )
                cluster_centers_indices = clus.cluster_centers_indices_

                n_clusters_ = len(cluster_centers_indices)
                for n in range(0, n_clusters_):
                    temp = []
                    tempf = []
                    tempKappa = []
                    print("subfeature: ", n)
                    for j in range(0, len(sub_feat[i])):
                        if clus.labels_[j] == n:
                            # to get the kappa
                            indexx = allorderdFe[i].index(sub_feat[i][j])
                            cc = np.argsort(allKappaTest[i])[::-1]
                            # finish
                            if len(allKappaTest[i]) != 0:
                                print(
                                    sub_feat[i][j], allKappaTest[i][cc[indexx]], "***"
                                )

                            tempf.append(sub_feat[i][j])
                            tempKappa.append(allKappaTest[i][cc[indexx]])
                            temp.append(self.__fea.index(sub_feat[i][j]))
                    print()
                    cluster2stLevel_index.append(temp)
                    clusterSubFeatuer.append(tempf)
                    clusterSubFeatuer_kappa.append(tempKappa)
                self.__cluster2stLevel_index_all.append(cluster2stLevel_index)
                self.__clusterSubFeatuer_all.append(clusterSubFeatuer)
                self.__clusterSubFeatuer_all_kappa.append(clusterSubFeatuer_kappa)

    def learning_sub_direction(self):
        # extracting the labels of the subfeatuers
        ##############################################################################
        ##############################################################################
        featuers2stLeve_Labeles_all = []

        cluster2stLeve_Labeles_all = []
        for cl in self.__cluster2stLevel_index_all:
            featuers2stLeve_Labeles = []
            for i in cl:
                temp = []
                for index in i:
                    temp.append(self.__train_labels[:, index])
                featuers2stLeve_Labeles.append(np.array(temp))
            featuers2stLeve_Labeles_all.append(np.array(featuers2stLeve_Labeles))
            # create label for each cluster
            cluster2stLeve_Labeles = []
            for i in featuers2stLeve_Labeles:
                j = np.sum(i, axis=0)
                # print(j)
                for k in range(0, len(j)):
                    if j[k] >= 1:
                        j[k] = 1
                cluster2stLeve_Labeles.append(j)
            cluster2stLeve_Labeles_all.append(cluster2stLeve_Labeles)
        print("step 13 extracting the labels of the subfeatuers")

        # In[ ]:

        # count the number of the subfeatuer and the primary
        print("we have ", len(self.__primaryFDirections), "primary features")
        count = len(self.__primaryFDirections)
        for i in self.__clusterSubFeatuer_all:

            count = count + len(i)
        print("WE Have ", count, "total features")

        # In[ ]:

        # learning the directions of the subfeatuers(in the postive part of the space)
        ##############################################################################
        ##############################################################################
        Sub_cluster_directions_ALL = []
        Sub_cluster_predictions_All = []

        Sub_cluster_kappa_All = []
        Sub_cluster_kappa_test_All = []

        for c in range(0, len(cluster2stLeve_Labeles_all)):
            print("cluster#", c, ":", self.__primarfeatures[c])

            Sub_cluster_directions = []
            Sub_cluster_predictions = []
            Sub_cluster_kappa = []
            Sub_cluster_kappa_test = []
            num = 0
            for i in cluster2stLeve_Labeles_all[c]:
                tempI = []
                for j in self.__primaryFPosIns[c]:
                    tempI.append(int(i[j]))

                (
                    kappa_scoretest,
                    kappa_score1,
                    f1,
                    direction,
                    acc,
                    TP,
                    FP,
                    TN,
                    FN,
                    predicted,
                    probabilities,
                ) = runLR1(self.__prim_PositveIns_vectors[c], np.array(tempI))
                Sub_cluster_directions.append(direction)
                Sub_cluster_predictions.append(predicted)

                Sub_cluster_kappa.append(kappa_score1)
                Sub_cluster_kappa_test.append(kappa_scoretest)
                print("sub_F# ", num, ":", self.__clusterSubFeatuer_all[c][num])

                num = num + 1
                del (
                    kappa_score1,
                    f1,
                    direction,
                    acc,
                    TP,
                    FP,
                    TN,
                    FN,
                    predicted,
                    probabilities,
                )
                gc.collect()

            print("---------------------------------------------")
            Sub_cluster_directions_ALL.append(Sub_cluster_directions)
            Sub_cluster_predictions_All.append(Sub_cluster_predictions)

            Sub_cluster_kappa_All.append(Sub_cluster_kappa)
            Sub_cluster_kappa_test_All.append(Sub_cluster_kappa_test)
        print(
            "step14 learning the directions of the subfeatuers (in the postive part of the space)"
        )

        # learning the orthogonal directions of the subfeatuers
        ##############################################################################
        ##############################################################################

        Sub_cluster_directions_ALL_Orthog = []
        print("len(self.__primaryFDirections)", len(self.__primaryFDirections))
        for i in range(0, len(self.__primaryFDirections)):
            temp = []
            # try:
            for j in range(0, len(Sub_cluster_directions_ALL[i])):
                # print(i,j)
                orthvect = []

                orthvect = (
                    Sub_cluster_directions_ALL[i][j]
                    - np.array(
                        (
                            (
                                np.dot(
                                    Sub_cluster_directions_ALL[i][j],
                                    self.__primaryFDirections[i],
                                )
                            )
                            / (
                                np.dot(
                                    self.__primaryFDirections[i],
                                    self.__primaryFDirections[i],
                                )
                            )
                        )
                    )
                    * self.__primaryFDirections[i]
                )

                # check = cosine_similarity([orthvect], [self.__primaryFDirections[i]])
                temp.append(orthvect)
            Sub_cluster_directions_ALL_Orthog.append(temp)
            # except:
            # print(i,j)

        print("step15 orthogonality")
        return Sub_cluster_directions_ALL, Sub_cluster_directions_ALL_Orthog

    def __clusAff(self, directions, terms):

        """
        This finction will cluster the directions using AffinityPropagation algorithm.
        """
        # here we tunned the preference
        # preference=-200.18430850091582
        aff = AffinityPropagation().fit(np.array(directions))

        cluster_centers_indices = aff.cluster_centers_indices_
        # labels = aff.labels_
        print("The Affinity matrix median", np.median(aff.affinity_matrix_))

        n_clusters_ = len(cluster_centers_indices)

        features_inPrimary_index = []  # cluster1stLevel_index
        PrimaryfeatClusters = []  # clusterFeatuerRana
        features_inPrimary_originalFeatuerDirection = []
        for i in range(0, n_clusters_):

            temp = []
            tempf = []
            tempDirections = []
            print("Primary features: ", i)
            for j in range(0, len(terms)):
                if aff.labels_[j] == i:
                    print(terms[j])
                    tempf.append(terms[j])
                    tempDirections.append(directions[j])
                    temp.append(self.__fea.index(terms[j]))
            if len(tempf) > 1:
                features_inPrimary_index.append(temp)
                PrimaryfeatClusters.append(tempf)
                features_inPrimary_originalFeatuerDirection.append(tempDirections)

        (
            cluster_directions,
            positiveMovies_cluster,
            PositveIns_vectors,
            PositveIns_Labels,
            clusterPrediction,
        ) = self.__findingclusterDirections(
            PrimaryfeatClusters, terms, directions, features_inPrimary_index
        )

        return (
            PrimaryfeatClusters,
            cluster_directions,
            positiveMovies_cluster,
            PositveIns_vectors,
            PositveIns_Labels,
            features_inPrimary_index,
            clusterPrediction,
        )

    def __findingclusterDirections(
        self, clusters, terms, terms_Direc, features_index_forechCluster
    ):

        """
        This func. will lrearn the directions in the full space

        """

        PrimaryFeatures_labels = self.__findingclusterLabels(
            clusters, features_index_forechCluster
        )
        cluster_directions = []
        cluster_predictions = []
        cluster_kappa = []
        cluster_kappa_test = []
        for i in PrimaryFeatures_labels:

            (
                kappa_scoretest,
                kappa_score1,
                f1,
                direction,
                acc,
                TP,
                FP,
                TN,
                FN,
                predicted,
                probabilities,
            ) = runLR1(self.__mds, i)

            cluster_directions.append(direction)
            cluster_predictions.append(predicted)
            cluster_kappa.append(kappa_score1)
            cluster_kappa_test.append(kappa_scoretest)

            del (
                kappa_score1,
                f1,
                direction,
                acc,
                TP,
                FP,
                TN,
                FN,
                predicted,
                probabilities,
            )
            gc.collect()

        (
            positiveMovies_cluster,
            PositveInstances_vectors,
            PositveInstances_Labels,
        ) = self.__findingPositiveInstances(clusters, cluster_predictions)
        return (
            cluster_directions,
            positiveMovies_cluster,
            PositveInstances_vectors,
            PositveInstances_Labels,
            cluster_predictions,
        )

    def __findingPositiveInstances(self, clustersArray, clustersPrediction):

        # extracting the postive entities
        ##############################################################################
        ##############################################################################
        PositveInstancesvectors = []
        PositveInstancesLabels = []
        PositveInstances = []
        for movie_i in range(0, len(clustersArray)):
            # pmiForCluster=[]
            tempmds = []
            templabel = []

            # print((flat_list))
            PosIn = [
                x1
                for x1 in range(0, len(self.__mds))
                if clustersPrediction[movie_i][x1] == 1
            ]
            PositveInstances.append(PosIn)
            # print(len(positiveMovies_cluster[movie_i]),len(x))
            for j in PosIn:
                tempmds.append(self.__mds[j])
                templabel.append(self.__train_labels[j])
            PositveInstancesvectors.append(tempmds)
            PositveInstancesLabels.append(templabel)
        # pmiForCluster.append(pmiFinalT[j])
        return PositveInstances, PositveInstancesvectors, PositveInstancesLabels

    def __findingclusterLabels(self, clustersArray, fratures_inCluster_index):

        # extracting the labels of the Primary
        ##############################################################################
        ##############################################################################
        featuers1stLeve_Labeles = []
        for i in fratures_inCluster_index:
            temp = []
            for index in i:
                temp.append(self.__train_labels[:, index])
            featuers1stLeve_Labeles.append(temp)
        cluster1stLeve_Labeles = []
        for i in featuers1stLeve_Labeles:
            j = np.sum(i, axis=0)
            for k in range(0, len(j)):
                if j[k] >= 1:
                    j[k] = 1
            cluster1stLeve_Labeles.append(j)
        return cluster1stLeve_Labeles

    @property
    def facetsSpaces(self):
        if self.__10dim_facets is None:
            raise NotTrainedError("Need to train model first.")
        return self.__10dim_facets

    @property
    def featuresForFacets(self):
        if self.__features_for_each_facet1 is None:
            raise NotTrainedError("Need to train model first.")
        return self.__features_for_each_facet1

    @property
    def featuresDircForFacets(self):
        if self.__featuresDirection_for_each_facet is None:
            raise NotTrainedError("Need to train model first.")
        return self.__featuresDirection_for_each_facet

    @property
    def featuresGloveDircForFacets(self):
        if self.__featuresDirection_for_each_facet_glove is None:
            raise NotTrainedError("Need to train model first.")
        return self.__featuresDirection_for_each_facet_glove


def perf_measure(y_actual, y_hat):  # Get the true positives etc
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == 1 and y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] == 0:
            FP += 1
        if y_actual[i] == 0 and y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] == 1:
            FN += 1

    return TP, FP, TN, FN


def runLR1(vectors, classes):
    # Default is dual formulation, which is unusual. Balanced balances the classes
    if len(np.unique(classes)) == 1:
        class_weight = None
    else:
        class_weight = "balanced"

    clf = linear_model.LogisticRegression(class_weight=class_weight, dual=False)
    clf.fit(
        vectors, classes
    )  # ,sample_weight=squared_sample)  # All of the vectors and classes. No need for training data.
    direction = clf.coef_.tolist()[0]  # Get the direction
    predicted = clf.predict(vectors)
    predicted = predicted.tolist()  # Convert to list so we can calculate the scores
    probabilities = clf.predict_proba(vectors)

    f1 = f1_score(classes, predicted)
    kappa_score = cohen_kappa_score(classes, predicted)
    kappa_score_test = cohen_kappa_score(classes, predicted)
    acc = accuracy_score(classes, predicted)
    TP, FP, TN, FN = perf_measure(classes, predicted)  # Get the True positive, etc
    return (
        kappa_score_test,
        kappa_score,
        f1,
        direction,
        acc,
        TP,
        FP,
        TN,
        FN,
        predicted,
        probabilities,
    )


def finding_directions(
    fea, space, facet, featuresTerms, GloveVectors, train_labels, Path
):
    """
    This function will return the orthogonal basis of the facet

    Parameters
    ----------
    space: The vector space where we want to learn the directions
    facet: The facet's number or -1 if to learn the directions in the entire space
    featuresTerms: The terms that we want to learn the directions for
    GloveVectors: the pre_trained Glove vectors for all the features(not only theone in the cluster)
    train_labels: the labels to train the classifier
    Path: To detrmain where to save the files


    Return value
    ----------
    #1-orderd_features: list containsthe terms orderd based on kappa score)
    #2-orderd_features_directions: ndarray contains the direction for each term
    #3-orderd_features_directions_GLOVE:
    #each line is a concatenation between the term's direction and the term's pre-trained word embedding vectors
    #4-orderd_features_Kappa(Kappa scores for each direction)
    #5-orderd_features_positiveMovies (movies that classified by the linear classifier as positive)
    ##############################################################################
    ##############################################################################

    """

    base_folder_second = Path

    directionsLGR = []
    scores = []
    predictions = []
    kappa_Score_all = []
    kappa_Score_all_test = []
    probabilities_features = []
    for i in range(0, len(featuresTerms)):
        # clusterpmiW = []
        if i % 200 == 0:

            start_time = time.time()
        if i % 200 == 0:
            print("--- %s seconds ---" % (time.time() - start_time))

        if i % 1000 == 0:
            print("finished: ", i)

        (
            kappa_scoretest,
            kappa_score1,
            f1,
            direction,
            acc,
            TP,
            FP,
            TN,
            FN,
            predicted,
            probabilities,
        ) = runLR1(space, train_labels[:, fea.index(featuresTerms[i])])

        directionsLGR.append(direction)
        predictions.append(predicted)
        probabilities_features.append(probabilities)
        kappa_Score_all.append(kappa_score1)
        kappa_Score_all_test.append(kappa_scoretest)
        scores.append([f1, acc])

    # print('STEP4_directions')

    # sort the directions
    ##############################################################################
    ##############################################################################
    positiveMovies = []
    for i in range(0, len(featuresTerms)):
        temp = []
        for j in range(0, len(predictions[i])):
            if predictions[i][j] == 1:
                temp.append(j)
        positiveMovies.append(temp)

    # filtered_directions = []
    accuracy = []
    for i in range(0, len(featuresTerms)):
        accuracy.append(kappa_Score_all_test[i])
    orderd_accuracy_index = np.argsort(accuracy)[::-1]
    # print('finish sorting')

    # writing the directions after sorting
    ##############################################################################
    ##############################################################################
    orderd_features1 = []
    orderd_features_directions1 = []
    orderd_features_kappa = []
    orderd_features_directions_GLOVE1 = []
    orderd_features_positiveMovies = []
    orderd_features_predictions = []
    for j in orderd_accuracy_index:
        orderd_features1.append(featuresTerms[int(j)])
        orderd_features_kappa.append(kappa_Score_all_test[int(j)])
        orderd_features_directions1.append(directionsLGR[int(j)])
        orderd_features_directions_GLOVE1.append(
            np.concatenate(
                (
                    directionsLGR[int(j)],
                    np.array(GloveVectors[fea.index(featuresTerms[int(j)])]),
                )
            )
        )
        orderd_features_positiveMovies.append(positiveMovies[j])
        orderd_features_predictions.append(predictions[j])
    if facet == -1:  # learning the directions in the entire space
        write2dArray(
            orderd_features_positiveMovies,
            base_folder_second + "orderd_featuresPositiveReclean",
        )
        write1dArray(orderd_features1, base_folder_second + "orderd_featuresReclean")
        write2dArray(
            orderd_features_directions1,
            base_folder_second + "orderd_features_directionsReclean",
        )
        write2dArray(
            orderd_features_directions_GLOVE1,
            base_folder_second + "orderd_features_directions_GLOVEReclean",
        )
        write1dArray(
            orderd_features_kappa, base_folder_second + "orderd_features_kappaReclean"
        )
    else:
        write1dArray(
            orderd_features1,
            base_folder_second + "orderd_features_for_facet_" + str(facet),
        )
        write2dArray(
            orderd_features_directions1,
            base_folder_second + "orderd_features_directions_for_facet_" + str(facet),
        )
        write2dArray(
            orderd_features_directions_GLOVE1,
            base_folder_second
            + "orderd_features_directions_GLOVE_for_facet_"
            + str(facet),
        )
        write1dArray(
            orderd_features_kappa,
            base_folder_second + "orderd_features_kappa_for_facet_" + str(facet),
        )

    return (
        orderd_features1,
        orderd_features_directions1,
        orderd_features_directions_GLOVE1,
        orderd_features_kappa,
        orderd_features_positiveMovies,
        orderd_features_predictions,
    )


# In[ ]:


def import2dArray(file_name, file_type="f", return_sparse=False):
    if file_name[-4:] == ".npz":
        print("Loading sparse array")
        array = sp.load_npz(file_name)
        if return_sparse is False:
            array = array.toarray()
    elif file_name[-4:] == ".npy":
        print("Loading numpy array")
        array = np.load(file_name)  #
    else:
        with open(file_name, "r") as infile:
            if file_type == "i":
                array = [list(map(int, line.strip().split())) for line in infile]
            elif file_type == "f":
                array = [list(map(float, line.strip().split())) for line in infile]
            elif file_type == "discrete":
                array = [list(line.strip().split()) for line in infile]
                for dv in array:
                    for v in range(len(dv)):
                        dv[v] = int(dv[v][:-1])
            else:
                array = np.asarray([list(line.strip().split()) for line in infile])
        array = np.asarray(array)
    print("successful import", file_name)
    return array


def write1dArray(array, name):
    try:
        file = open(name, "w")
        print("starting array")
        for i in range(len(array)):
            file.write(str(array[i]))
            file.write("\n")
        file.close()
    except FileNotFoundError:
        print("FAILURE")

    print("successful write", name)


def write2dArray(array, name):
    try:
        file = open(name, "w")
        print("starting array")
        for i in range(len(array)):
            for n in range(len(array[i])):
                file.write(str(array[i][n]) + " ")
            file.write("\n")
        file.close()
    except FileNotFoundError:
        print("FAILURE")

    print("successful write", name)
