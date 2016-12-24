#! /bin/sh
#
# download_anno.sh
# Copyright (C) 2016 prithv1 <prithv1@vt.edu>
#
# Distributed under terms of the MIT license.
#


wget https://filebox.ece.vt.edu/~prithv1/vqa_jsons.tar.gz && tar -xvzf vqa_jsons.tar.gz && rm -rf vqa_jsons.tar.gz && cp $HOME/Cloud-CV/VQA-Evaluation/Results/OpenEnded_mscoco_train2014_fake_results.json VQA_jsons/
