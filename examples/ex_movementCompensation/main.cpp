//=============================================================================================================
/**
 * @file     main.cpp
 * @author   Ruben Dörfel <doerfelruben@aol.com>
 * @since    0.1.0
 * @date     06, 2020
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
 * @brief     Evaluation script for head movement compensation method.
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

#include <inverse/hpiFit/hpifit.h>

#include <utils/ioutils.h>
#include <utils/generics/applicationlogger.h>
#include <utils/mnemath.h>

#include <fwd/fwd_coil_set.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QtCore/QCoreApplication>
#include <QFile>
#include <QCommandLineParser>
#include <QDebug>
#include <QGenericMatrix>
#include <QElapsedTimer>

//=============================================================================================================
// Eigen
//=============================================================================================================

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace INVERSELIB;
using namespace FIFFLIB;
using namespace UTILSLIB;
using namespace Eigen;

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
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    QCoreApplication a(argc, argv);

    // Command Line Parser
    QCommandLineParser parser;
    parser.setApplicationDescription("Example name");
    parser.addHelpOption();

    QCommandLineOption parameterOption("parameter", "The first parameter description.");

    parser.addOption(parameterOption);

    parser.process(a);

    //=============================================================================================================
    // Load data
    QFile t_fileIn(QCoreApplication::applicationDirPath() + "/MNE-sample-data/simulate/sim-chpi-move-aud-raw.fif");
    FiffRawData rawData(t_fileIn);
    QSharedPointer<FiffInfo> pFiffInfo = QSharedPointer<FiffInfo>(new FiffInfo(rawData.info));
    //=============================================================================================================

    //=============================================================================================================
    // Setup customisable values
    // Data Stream
    fiff_int_t iQuantum = 200;              // Buffer size

    // HPI Fitting
    double dErrorMax = 0.10;                // maximum allowed estimation error
    double dAllowedMovement = 0.3;          // maximum allowed movement
    double dAllowedRotation = 0.5;          // maximum allowed rotation
    bool bDoFastFit = true;                 // fast fit or advanced linear model

    //=============================================================================================================
    // setup data stream
    MatrixXd matData, matTimes;
    MatrixXd matPosition;                   // matPosition matrix to save quaternions etc.

    fiff_int_t from, to;
    fiff_int_t first = rawData.first_samp;
    fiff_int_t last = rawData.last_samp;

    int iN = ceil((last-first)/iQuantum);   // number of data segments
    qDebug() << first << last << iN << iQuantum;
    MatrixXd vecTime = RowVectorXd::LinSpaced(iN, 0, iN-1)*iQuantum;
    //=============================================================================================================

    //=============================================================================================================
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

    //=============================================================================================================
    // setup informations for HPI fit (VectorView)
    QVector<int> vecFreqs {154,161,158,166};
    bool bSorted = false;
    QVector<double> vecError;
    VectorXd vecGoF;
    FiffDigPointSet fittedPointSet;

    // if debugging files are necessary set bDoDebug = true;
    QString sHPIResourceDir = QCoreApplication::applicationDirPath() + "/HPIFittingDebug";
    bool bDoDebug = false;

    // HPI values
    double dMeanErrorDist = 0.0;
    double dMovement = 0.0;
    double dRotation = 0.0;

    HpiFitResult fitResult;
    fitResult.devHeadTrans = pFiffInfo->dev_head_t;
    fitResult.devHeadTrans.from = 1;
    fitResult.devHeadTrans.to = 4;
    FiffCoordTrans transDevHeadRef = pFiffInfo->dev_head_t;
    FiffCoordTrans transDevHeadFit = pFiffInfo->dev_head_t;
    // create HPI fit object
    HPIFit HPI = HPIFit(pFiffInfo,bDoFastFit);
    //=============================================================================================================

    //=============================================================================================================
    // actual pipeline

    for(int i = 0; i < vecTime.size(); i++) {
        from = first + vecTime(i);
        to = from + iQuantum;
        if (to > last) {
            to = last;
            qWarning() << "Block size < iQuantum " << iQuantum;
        }
        // Reading
        if(!rawData.read_raw_segment(matData, matTimes, from, to)) {
            qCritical("error during read_raw_segment");
            return -1;
        }

        if(!bSorted) {
            while(!bSorted) {
                qDebug() << "order coils";
                bSorted = HPI.findOrder(matData,
                                        matProjectors,
                                        transDevHeadFit,
                                        vecFreqs,
                                        fitResult.errorDistances,
                                        fitResult.GoF,
                                        fitResult.fittedCoils,
                                        pFiffInfo);
            }
            qDebug() << "Coil frequencies ordered: " << vecFreqs;
        }

        qInfo() << "HPI-Fit...";
        HPI.fitHPI(matData,
                   matProjectors,
                   transDevHeadFit,
                   vecFreqs,
                   fitResult.errorDistances,
                   fitResult.GoF,
                   fitResult.fittedCoils,
                   pFiffInfo,
                   bDoDebug,
                   sHPIResourceDir);
        qInfo() << "[done]";
        // get error
        dMeanErrorDist = std::accumulate(fitResult.errorDistances.begin(), fitResult.errorDistances.end(), .0) / fitResult.errorDistances.size();
        // update head position
        if(dMeanErrorDist < dErrorMax) {
            HPIFit::storeHeadPosition(first/pFiffInfo->sfreq, fitResult.devHeadTrans.trans, matPosition, fitResult.GoF, fitResult.errorDistances);
            fitResult.devHeadTrans = transDevHeadFit;
            // check displacement
            dMovement = transDevHeadRef.translationTo(fitResult.devHeadTrans.trans);
            dRotation = transDevHeadRef.angleTo(fitResult.devHeadTrans.trans);
            qDebug() << dMovement;
            qDebug() << dRotation;
            if(dMovement > dAllowedMovement || dRotation > dAllowedRotation) {
                matPosition(i,9) = 1;
                fitResult.bIsLargeHeadMovement = true;              // update head position
                transDevHeadRef = fitResult.devHeadTrans;
                qDebug() << "big head movement";
            }
        }

    }
    return a.exec();
}
