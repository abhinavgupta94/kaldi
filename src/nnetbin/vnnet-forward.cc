// nnetbin/nnet-forward.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include <sstream>
#include <string>
#include <iostream>


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform forward pass through Neural Network.\n"
        "\n"
        "Usage:  vnnet-forward [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " vnnet-forward nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //int32 time_shift = 0;

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    // optionally remove softmax,
    Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
    if (no_softmax) {
      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
        nnet.RemoveComponent(nnet.NumComponents()-1);
      } else {
        KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
      }
    }

    // avoid some bad option combinations,
    if (apply_log && no_softmax) {
      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
    }

    // we will subtract log-priors later,
    PdfPrior pdf_prior(prior_opts); 

    // disable dropout,
    nnet_transf.SetDropoutRetention(1.0);
    nnet.SetDropoutRetention(1.0);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;


    Timer time;
    double time_now = 0;
    int32 num_done = 0;
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::string utt = feature_reader.Key();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << utt
                    << ", " << mat.NumRows() << "frm";

      
      if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }

      // push it to gpu,
      feats = mat;

      // fwd-pass, feature transform,
      nnet_transf.Feedforward(feats, &feats_transf);
      if (!KALDI_ISFINITE(feats_transf.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in transformed-features for " << utt;
      }

      // fwd-pass, nnet, for visualization
      nnet.vFeedforward(feats_transf, &nnet_out);
      std::vector<CuMatrix<BaseFloat> > all_activations = nnet.PropagateBuffer();
      //the length of this vector is NumComponents() + 1. Dont know why they put it like that

      //verifying the final output
      int32 noa = all_activations.size();

      for (int i = 1; i < noa - 1; i = i + 2) {
        //skip the linear part and use only the sigmoid activation part
        //checking for nan/inf

        if (!KALDI_ISFINITE(all_activations[i].Sum())) {
          KALDI_ERR << "NaN or inf found in nn-activations for " << utt;
        }

        // download from GPU,
        nnet_out_host.Resize(all_activations[i].NumRows(), all_activations[i].NumCols());
        //nnet_out.CopyToMat(&nnet_out_host);
        all_activations[i].CopyToMat(&nnet_out_host);

        std::string output_key = feature_reader.Key();


        if ( i != noa - 2) {
          int layer_counter = (i/2) + 1;
          std::ostringstream layerNo;
          layerNo << layer_counter;
          output_key = output_key + "-HLAYER" + layerNo.str();
        }

        feature_writer.Write(output_key, nnet_out_host);


      }

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
