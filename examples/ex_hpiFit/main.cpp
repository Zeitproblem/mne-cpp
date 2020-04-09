//=============================================================================================================
/**
 * @file     main.cpp
 * @author   Ruben Dörfel <ruben.doerfel@tu-ilmenau.de>;
 * @since    0.1.0
 * @date     January, 2020
 *
 * @section  LICENSE
 *
 * Copyright (C) 2020, Ruben Dörfel. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 * the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 *       the following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of MNE-CPP authors nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * @brief     Example for cHPI fitting on raw data with SSP. The result is written to a .txt file for comparison with MaxFilter's .pos file.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <iostream>
#include <vector>

#include <fiff/fiff.h>
#include <fiff/fiff_info.h>
#include <fiff/fiff_dig_point_set.h>

#include <fwd/fwd_coil_set.h>

#include <inverse/hpiFit/hpifit.h>

#include <mne/mne.h>

#include <utils/ioutils.h>
#include <utils/generics/applicationlogger.h>
#include <utils/mnemath.h>

//=============================================================================================================
// Qt INCLUDES
//=============================================================================================================

#include <QtCore/QCoreApplication>
#include <QFile>
#include <QCommandLineParser>
#include <QDebug>
#include <QGenericMatrix>
#include <QElapsedTimer>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace INVERSELIB;
using namespace FIFFLIB;
using namespace UTILSLIB;
using namespace Eigen;
using namespace MNELIB;

//=============================================================================================================
// MAIN
//=============================================================================================================

//=============================================================================================================
/**
 * The function main marks the entry point of the program.
 * By default, main has the storage class extern.
 *
 * @param [in] argc (argument count) is an integer that indicates how many arguments were entered on the command line when the program was started.
 * @param [in] argv (argument vector) is an array of pointers to arrays of character objects. The array objects are null-terminated strings, representing the arguments that were entered on the command line when the program was started.
 * @return the value that was set to exit() (which is 0 if exit() is called via quit()).
 */

int main(int argc, char *argv[])
{
    qInstallMessageHandler(ApplicationLogger::customLogWriter);
    QElapsedTimer timer;
    QCoreApplication a(argc, argv);

    // Command Line Parser
    QCommandLineParser parser;
    parser.setApplicationDescription("hpiFit Example");
    parser.addHelpOption();
    qInfo() << "Please download the mne-cpp-test-data folder from Github (mne-tools) into mne-cpp/bin.";
    QCommandLineOption inputOption("fileIn", "The input file <in>.", "in", QCoreApplication::applicationDirPath() + "/MNE-sample-data/chpi/raw/BabyMeg_170307_104229_baby_doll_01_raw.fif");
    QCommandLineOption digitOption("fileIn", "The input file <in>.", "in", QCoreApplication::applicationDirPath() + "/MNE-sample-data/chpi/raw/BabyMeg_fastscan_20170202_babydoll_set1.fif");

    parser.addOption(inputOption);
    parser.addOption(digitOption);
    parser.process(a);

    // Init data loading and writing
    QFile t_fileIn(parser.value(inputOption));
    QFile t_fileDigit(parser.value(digitOption));
    FiffRawData raw(t_fileIn);

    FiffDigPointSet digSet(t_fileDigit);

    bool keep_comp = false;
    int dest_comp = 101;

    MNE::setup_compensators(raw,
                            dest_comp,
                            keep_comp);

    QSharedPointer<FiffInfo> pFiffInfo = QSharedPointer<FiffInfo>(new FiffInfo(raw.info));

    QList<FiffDigPoint> lDigSet;

    for(int i = 0; i < digSet.size(); ++i) {
        switch(digSet[i].kind) {
            case FIFFV_POINT_HPI: {
                lDigSet.append(digSet[i]);
            }
        }
    }

    pFiffInfo->dig = lDigSet;

    // Setup comparison of transformation matrices
    FiffCoordTrans transDevHead = pFiffInfo->dev_head_t;    // transformation that only updates after big head movements
    float fThreshRot = 5;          // in degree
    float fThreshTrans = 0.005;    // in m

    // Set up the reading parameters
    RowVectorXi vecPicks = pFiffInfo->pick_types(true, false, false);

    MatrixXd matData, matTimes;

    fiff_int_t from, to;
    fiff_int_t first = raw.first_samp;
    fiff_int_t last = raw.last_samp;

    float fQuantumSec = 0.2f;       // read and write in 200 ms junks
    fiff_int_t iQuantum = ceil(fQuantumSec*pFiffInfo->sfreq);

    // create time vector that specifies when to fit
    float dTSec = 0.1;                              // time between hpi fits
    int iQuantumT = floor(dTSec*pFiffInfo->sfreq);   // samples between fits
    int iN = floor((last-first)/iQuantumT);
    RowVectorXf vecTime = RowVectorXf::LinSpaced(iN, 0, iN-10) * dTSec;

    // To fit at specific times outcommend the following block
    // Read Quaternion File
//    MatrixXd matPos;
//    qInfo() << "Specify the path to your Position file (.txt)";
//    IOUtils::read_eigen_matrix(matPos, QCoreApplication::applicationDirPath() + "/mne-cpp-test-data/Result/ref_hpiFit_pos.txt");
//    RowVectorXd vecTime = matPos.col(0);

    MatrixXd matPosition;              // matPosition matrix to save quaternions etc.

    // setup informations for HPI fit (VectorView)
    QVector<int> vecFreqs {155,165,190,220};
    QVector<double> vecError;
    double dError = 0;
    VectorXd vecGoF;
    FiffDigPointSet fittedPointSet;

    // Setup Comps
    MatrixXd matComp = MatrixXd::Identity(pFiffInfo->chs.size(), pFiffInfo->chs.size());

    FiffCtfComp newComp;
    //Do this always from 0 since we always read new raw data, we never actually perform a multiplication on already existing data
    if(pFiffInfo->make_compensator(0, 101, newComp)) {
        matComp = newComp.data->data;
    }

    // Use SSP + SGM + calibration
    MatrixXd matProjectors = MatrixXd::Identity(pFiffInfo->chs.size(), pFiffInfo->chs.size());

    //Do a copy here because we are going to change the activity flags of the SSP's
    FiffInfo infoTemp = *(pFiffInfo.data());

    //Turn on all SSP
    for(int i = 0; i < infoTemp.projs.size(); ++i) {
        infoTemp.projs[i].active = true;
    }

    //Create the projector for all SSP's on
    infoTemp.make_projector(matProjectors);

    //set columns of matrix to zero depending on bad channels indexes
    for(qint32 j = 0; j < infoTemp.bads.size(); ++j) {
        matProjectors.col(infoTemp.ch_names.indexOf(infoTemp.bads.at(j))).setZero();
    }

    // if debugging files are necessary set bDoDebug = true;
    QString sHPIResourceDir = QCoreApplication::applicationDirPath() + "/HPIFittingDebug";
    bool bDoDebug = false;
    bool bDoFastFit = true;

    HPIFit HPI = HPIFit(pFiffInfo,bDoFastFit);

    float fTimer = 0.0;

    // read and fit
    for(int i = 0; i < vecTime.size(); i++) {
        from = first + vecTime(i)*pFiffInfo->sfreq;
        to = from + iQuantum;
        if (to > last) {
            to = last;
            qWarning() << "Block size < iQuantum " << iQuantum;
        }
        // Reading
        if(!raw.read_raw_segment(matData, matTimes, from, to)) {
            qCritical("error during read_raw_segment");
            return -1;
        }
        qInfo() << "[done]";

        transDevHead = pFiffInfo->dev_head_t;

        qInfo() << "HPI-Fit...";
        timer.start();
        HPI.fitHPI(matData,
                   matProjectors,
                   transDevHead,
                   vecFreqs,
                   vecError,
                   vecGoF,
                   fittedPointSet,
                   pFiffInfo,
                   bDoDebug,
                   sHPIResourceDir);
        fTimer = timer.elapsed();
        qInfo() << "The HPI-Fit took" << fTimer << "milliseconds";
        qInfo() << "[done]";

        HPIFit::storeHeadPosition(vecTime(i), pFiffInfo->dev_head_t.trans, matPosition, vecGoF, vecError);
        matPosition(i,9) = fTimer;

        // only update transformation matrix if error is smaller then threshold. otherwise use old one. 
        dError = std::accumulate(vecError.begin(), vecError.end(), .0) / vecError.size();
        if(dError < 0.010) {
            pFiffInfo->dev_head_t = transDevHead;
        } else {
            qInfo() << "Large error.";
        }
    }
    IOUtils::write_eigen_matrix(matPosition, QCoreApplication::applicationDirPath() + "/MNE-sample-data/chpi/pos/pos_00_BabyMeg_Home.txt");
}
