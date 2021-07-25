function [jaccard] = jaccard_metric(salient_img,GT_img)
jaccard= sum(salient_img & GT_img)/sum(salient_img | GT_img);