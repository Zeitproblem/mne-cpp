//=============================================================================================================
/**
 * @file     main.cpp
 * @author   Ruben Dörfel <doerfelruben@aol.com>
 * @since    0.1.0
 * @date     June, 2020
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
#include <fiff/fiff_ch_info.h>
#include <fiff/c/fiff_digitizer_data.h>
#include <fiff/fiff_cov.h>
#include <fiff/fiff_evoked_set.h>

#include <inverse/hpiFit/hpifit.h>
#include <inverse/minimumNorm/minimumnorm.h>

#include <utils/ioutils.h>
#include <utils/generics/applicationlogger.h>
#include <utils/mnemath.h>

#include <fwd/fwd_coil_set.h>
#include <fwd/computeFwd/compute_fwd.h>
#include <fwd/computeFwd/compute_fwd_settings.h>

#include <disp3D/viewers/abstractview.h>
#include <disp3D/engine/model/data3Dtreemodel.h>

#include <fs/surfaceset.h>
#include <fs/annotationset.h>

#include <disp/viewers/control3dview.h>
#include <disp/viewers/butterflyview.h>
#include <disp/viewers/helpers/evokedsetmodel.h>
#include <disp/viewers/helpers/channelinfomodel.h>
#include <disp/plots/imagesc.h>

#include <disp3D/viewers/networkview.h>
#include <disp/viewers/minimumnormsettingsview.h>
#include <disp3D/engine/model/items/network/networktreeitem.h>
#include <disp3D/engine/model/items/sourcedata/mnedatatreeitem.h>
#include <disp3D/engine/model/items/sensorspace/sensorsettreeitem.h>
#include <disp3D/engine/model/data3Dtreemodel.h>
#include <disp3D/engine/model/items/freesurfer/fssurfacetreeitem.h>
#include <disp3D/engine/view/view3D.h>
#include <disp3D/engine/model/data3Dtreemodel.h>
#include <disp3D/engine/delegate/data3Dtreedelegate.h>
#include <disp3D/engine/model/items/bem/bemtreeitem.h>
#include <disp3D/engine/model/items/bem/bemsurfacetreeitem.h>
#include <disp3D/engine/model/items/digitizer/digitizertreeitem.h>
#include <disp3D/engine/model/items/digitizer/digitizersettreeitem.h>

#include <mne/c/mne_msh_display_surface_set.h>
#include <mne/c/mne_surface_or_volume.h>
#include <mne/mne_forwardsolution.h>
#include <mne/mne_sourceestimate.h>
#include <mne/mne.h>

#include <rtprocessing/filter.h>
#include <rtprocessing/helpers/filterkernel.h>

#include <rtprocessing/rtcov.h>
#include <rtprocessing/rtinvop.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QApplication>
#include <QtCore/QCoreApplication>
#include <QFile>
#include <QCommandLineParser>
#include <QDebug>
#include <QGenericMatrix>
#include <QElapsedTimer>
#include <QAction>
#include <Qt3DCore/QTransform>
#include <QPointer>

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
using namespace DISP3DLIB;
using namespace FSLIB;
using namespace MNELIB;
using namespace FWDLIB;
using namespace RTPROCESSINGLIB;

//=============================================================================================================
// member variables
//=============================================================================================================

Eigen::SparseMatrix<double> updateProjectors(FiffInfo infoTemp) {

    // Use SSP + SGM + calibration
    MatrixXd matProjectors = MatrixXd::Identity(infoTemp.chs.size(), infoTemp.chs.size());

    //Turn on all SSP
    for(int i = 0; i < infoTemp.projs.size(); ++i) {
        infoTemp.projs[i].active = true;
    }

    //Create the projector for all SSP's on
    infoTemp.make_projector(matProjectors);

    qint32 nchan = infoTemp.nchan;
    qint32 i, k;

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(nchan);

    tripletList.clear();
    tripletList.reserve(matProjectors.rows()*matProjectors.cols());
    for(i = 0; i < matProjectors.rows(); ++i) {
        for(k = 0; k < matProjectors.cols(); ++k) {
            if(matProjectors(i,k) != 0) {
                tripletList.push_back(T(i, k, matProjectors(i,k)));
            }
        }
    }

    //set columns of matrix to zero depending on bad channels indexes
    for(qint32 j = 0; j < infoTemp.bads.size(); ++j) {
        matProjectors.col(infoTemp.ch_names.indexOf(infoTemp.bads.at(j))).setZero();
    }

    Eigen::SparseMatrix<double> matSparseProjMult = SparseMatrix<double>(matProjectors.rows(),matProjectors.cols());
    if(tripletList.size() > 0)
        matSparseProjMult.setFromTriplets(tripletList.begin(), tripletList.end());

    return matSparseProjMult;
}

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
    QApplication a(argc, argv);

    // Command Line Parser
    QCommandLineParser parser;
    parser.setApplicationDescription("Evaluation for movement compensation");
    parser.addHelpOption();

    QCommandLineOption moveCompOption("movComp", "Perform head position updates", "bool", "false");
    QCommandLineOption hpiOption("hpi", "Is HPI fitting acquired?", "bool", "true");
    QCommandLineOption fastFitOption("fastFit", "Do fast HPI fits.", "bool", "true");
    QCommandLineOption maxMoveOption("maxMove", "Maximum head movement in mm.", "value", "3");
    QCommandLineOption maxRotOption("maxRot", "Maximum head rotation in degree.", "value", "5");
    QCommandLineOption coilTypeOption("coilType", "The coil <type> (for sensor level usage only), 'eeg', 'grad' or 'mag'.", "type", "grad");
    QCommandLineOption clustOption("cluster", "Do clustering of source space (for source level usage only).", "doClust", "true");
    QCommandLineOption rawFileOption("fileIn", "The input file <in>.", "in", QCoreApplication::applicationDirPath() + "/MNE-sample-data/chpi/simulation/sim-chpi-move-aud-raw.fif");
    QCommandLineOption logOption("log", "Log the computation and store results to file.", "bool", "true");
    QCommandLineOption logDirOption("logDir", "The directory to log the results.", "string", QCoreApplication::applicationDirPath() + "/MNE-sample-data/chpi/evaluation");
    QCommandLineOption idOption("id", "The id for the measurement", "id", "id");
    QCommandLineOption eventsFileOption("events", "Path to the event <file>.", "file", QCoreApplication::applicationDirPath() + "/MNE-sample-data/chpi/simulation/sim-aud-eve.fif");

    parser.addOption(moveCompOption);
    parser.addOption(hpiOption);
    parser.addOption(fastFitOption);
    parser.addOption(maxMoveOption);
    parser.addOption(maxRotOption);
    parser.addOption(coilTypeOption);
    parser.addOption(clustOption);
    parser.addOption(rawFileOption);
    parser.addOption(logOption);
    parser.addOption(logDirOption);
    parser.addOption(idOption);
    parser.addOption(eventsFileOption);

    parser.process(a);

    //=============================================================================================================
    // Setup customisable values via command line

    // Data Stream
    QString sRaw = parser.value(rawFileOption);
    QFile fileEvent(parser.value(eventsFileOption));

    // HPI Fitting
    float dAllowedMovement = parser.value(maxMoveOption).toFloat()/1000.0;
    float dAllowedRotation = parser.value(maxRotOption).toFloat();
    bool bDoFastFit = true;                 // fast fit or advanced linear model
    bool bDoHPIFit = true;

    if(parser.value(fastFitOption) == "false" || parser.value(fastFitOption) == "0") {
        bDoFastFit = false;
    } else if(parser.value(fastFitOption) == "true" || parser.value(fastFitOption) == "1") {
        bDoFastFit = true;
    }
    if(parser.value(hpiOption) == "false" || parser.value(hpiOption) == "0") {
        bDoHPIFit = false;
    } else if(parser.value(hpiOption) == "true" || parser.value(hpiOption) == "1") {
        bDoHPIFit = true;
    }

    // Forward Solution
    bool bDoHeadPosUpdate = false;
    bool bDoCluster = true;                 // not implemented, always cluster

    if(parser.value(moveCompOption) == "false" || parser.value(moveCompOption) == "0") {
        bDoHeadPosUpdate = false;
    } else if(parser.value(moveCompOption) == "true" || parser.value(moveCompOption) == "1") {
        bDoHeadPosUpdate = true;
    }

    if(parser.value(clustOption) == "false" || parser.value(clustOption) == "0") {
        bDoCluster = false;
    } else if(parser.value(clustOption) == "true" || parser.value(clustOption) == "1") {
        bDoCluster = true;
    }

    // Sensor Level
    QString sCoilType = parser.value(coilTypeOption);

    // log option
    QString sLogDir = parser.value(logDirOption);
    QString sID = parser.value(idOption);

    bool bDoLogging = false;
    if(parser.value(logOption) == "false" || parser.value(logOption) == "0") {
        bDoLogging = false;
    } else if(parser.value(logOption) == "true" || parser.value(logOption) == "1") {
        bDoLogging = true;
    }

    //=============================================================================================================
    // Set data paths
    QFile t_fileMriName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/MEG/sample/all-trans.fif");
    QFile t_fileBemName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem.fif");
    QFile t_fileSrcName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/bem/sample-oct-6-src.fif");
    QFile t_fileMeasName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/MEG/sample/sample_audvis_raw.fif");
    QFile t_fileCov(QCoreApplication::applicationDirPath() + "/MNE-sample-data/MEG/sample/sample_audvis-cov.fif");

    QString sAtlasDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/label");
    QString sSurfaceDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/surf");

    //=============================================================================================================
    // Load data

    QFile fileRaw(sRaw);
    FiffRawData::SPtr pRawData = FiffRawData::SPtr::create(fileRaw);
    FiffRawData rawMeas(t_fileMeasName);

    // Setup compensators and projectors so they get applied while reading
    bool bKeepComp = false;
    fiff_int_t iDestComp = 0;
    MNE::setup_compensators(*pRawData.data(),
                            iDestComp,
                            bKeepComp);

    FiffInfo::SPtr pFiffInfo = FiffInfo::SPtr::create(pRawData->info);

    QStringList lExclude;
    lExclude << pRawData->info.bads << pRawData->info.ch_names.filter("EOG") << "MEG2641" << "MEG2443" << "EEG053";

    RowVectorXi vecPicks;
    // pick sensor types
    if(sCoilType.contains("grad", Qt::CaseInsensitive)) {
        vecPicks = pRawData->info.pick_types(QString("grad"),false,false,QStringList(),lExclude);
    } else if (sCoilType.contains("mag", Qt::CaseInsensitive)) {
        vecPicks = pRawData->info.pick_types(QString("mag"),false,false,QStringList(),lExclude);
    } else if (sCoilType.contains("eeg", Qt::CaseInsensitive)) {
        vecPicks = pRawData->info.pick_types(QString("all"),true,false,QStringList(),lExclude);
    } else {
        vecPicks = pRawData->info.pick_types(QString("all"),false,false,QStringList(),lExclude);
    }

    FiffInfo::SPtr pFiffInfoCompute = FiffInfo::SPtr::create(pFiffInfo->pick_info(vecPicks));
    QStringList lPickedChannels = pFiffInfoCompute->ch_names;

    // further loading
    AnnotationSet::SPtr pAnnotationSet = AnnotationSet::SPtr::create(sAtlasDir+"/lh.aparc.a2009s.annot", sAtlasDir+"/rh.aparc.a2009s.annot");
    FiffCoordTrans mriHeadTrans(t_fileMriName); // mri <-> head transformation matrix

    //=============================================================================================================
    // setup data stream

    MatrixXd matData, matTimes;
    MatrixXd matPosition;                   // matPosition matrix to save quaternions etc.

    fiff_int_t iQuantum = 180;              // Buffer size
    fiff_int_t from, to;
    fiff_int_t first = pRawData->first_samp;
    fiff_int_t last = pRawData->last_samp;

    int iN = ceil((last-first)/iQuantum);   // number of data segments
    VectorXd vecTime = RowVectorXd::LinSpaced(iN, 0, iN-1)*iQuantum;

    Eigen::SparseMatrix<double> matSparseProjMult = updateProjectors(*(pFiffInfo.data()));

    //=============================================================================================================
    // setup Covariance

    FiffCov noiseCov(t_fileCov);
    noiseCov = noiseCov.regularize(*pFiffInfoCompute.data(), 0.05, 0.05, 0.1, true);

    //=============================================================================================================
    // setup informations for HPI fit (VectorView)

    QVector<int> vecFreqs {166,154,161,158};
    bool bSorted = false;
    QVector<double> vecError;
    VectorXd vecGoF;
    FiffDigPointSet fittedPointSet;

    // if debugging files are necessary set bDoDebug = true;
    QString sHPIResourceDir = QCoreApplication::applicationDirPath() + "/HPIFittingDebug";
    bool bDoDebug = false;

    // HPI values
    double dErrorMax = 0.010;                // maximum allowed estimation error

    double dMeanErrorDist = 0.0;
    double dMovement = 0.0;
    double dRotation = 0.0;

    HpiFitResult fitResult;
    fitResult.devHeadTrans = pFiffInfo->dev_head_t;
    fitResult.devHeadTrans.from = 1;
    fitResult.devHeadTrans.to = 4;
    FiffCoordTrans transDevHeadRef = pFiffInfo->dev_head_t;     // reference of last fit
    FiffCoordTrans transDevHeadFit = pFiffInfo->dev_head_t;
    // create HPI fit object
    pFiffInfo->linefreq = 60.0;
    HPIFit HPI = HPIFit(pFiffInfo,bDoFastFit);

    //=============================================================================================================
    // setup Forward solution

    QFile t_fileAtlasDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/label");
    VectorXd vecTimeUpdate = vecTime;       // store time elapsed during head position update
    if(!bDoHeadPosUpdate) {
        vecTimeUpdate.fill(0);
    }

    ComputeFwdSettings::SPtr pFwdSettings = ComputeFwdSettings::SPtr::create();
    pFwdSettings->solname = QCoreApplication::applicationDirPath() + "/MNE-sample-data/your-solution-fwd.fif";
    pFwdSettings->mriname = t_fileMriName.fileName();
    pFwdSettings->bemname = t_fileBemName.fileName();
    pFwdSettings->srcname = t_fileSrcName.fileName();
    pFwdSettings->measname = t_fileMeasName.fileName();
    pFwdSettings->transname.clear();
    pFwdSettings->eeg_model_name = "Default";
    pFwdSettings->include_meg = true;
    pFwdSettings->include_eeg = true;
    pFwdSettings->accurate = true;
    pFwdSettings->mindist = 5.0f/1000.0f;
    pFwdSettings->pFiffInfo = pFiffInfoCompute;

    FiffCoordTransOld transMegHeadOld = pFiffInfoCompute->dev_head_t.toOld();
    QFile t_fSolution(pFwdSettings->solname);

    MNEForwardSolution::SPtr pFwdSolution;
    MNEForwardSolution::SPtr pClusteredFwd;

    // init Forward solution
    ComputeFwd::SPtr pComputeFwd = ComputeFwd::SPtr::create(pFwdSettings);

    // compute and store first forward solution
    pComputeFwd->calculateFwd();
    pComputeFwd->storeFwd();

    Eigen::MatrixXd defaultD;       // default cluster operator

    pFwdSolution = MNEForwardSolution::SPtr::create(t_fSolution);
    pClusteredFwd = MNEForwardSolution::SPtr::create(pFwdSolution->cluster_forward_solution(*pAnnotationSet.data(), 20, defaultD, noiseCov));

    MNEForwardSolution forwardCompute;

    if(bDoCluster) {
        forwardCompute  = pClusteredFwd->pick_channels(lPickedChannels);
    } else {
        forwardCompute  = pFwdSolution->pick_channels(lPickedChannels);
    }

    //=============================================================================================================
    // read events

    // Read the events
    MatrixXi matEvents;
    MNE::read_events(fileEvent,
                     matEvents);

    //=============================================================================================================
    // Logging

    qInfo() << "-------------------------------------------------------------------------------------------------";
    qInfo() << "File: " << sRaw;
    qInfo() << "Coil Type: " << sCoilType;
    qInfo() << "Fast HPI Fit: " << bDoFastFit;
    qInfo() << "Movement compensation: " << bDoHeadPosUpdate;
    qInfo() << "Maximum rotation: " << dAllowedRotation;
    qInfo() << "Maximum displacement" << dAllowedMovement;
    qInfo() << "Sources" << forwardCompute.nsource;
    qInfo() << "-------------------------------------------------------------------------------------------------";

    QString sCurrentDir;
    if(bDoLogging) {

        QString sTimeStamp = QDateTime::currentDateTime().toString("yyMMdd_hhmmss");

        if(!QDir(sLogDir).exists()) {
            QDir().mkdir(sLogDir);
        }

        sCurrentDir = sLogDir + "/" + sTimeStamp + "_" + sID;
        QDir().mkdir(sCurrentDir);

        pComputeFwd->storeFwd(sCurrentDir + "/" + "sim-aud-" + sCoilType + "-fwd.fif");

        QString sLogFile = sID + "_LogFile.txt";
        QFile file(sCurrentDir + "/" + sLogFile);

        if(file.open(QIODevice::WriteOnly|QIODevice::Truncate)) {
            QTextStream stream(&file);
            stream << "ex_movementCompensation" << "\n";
            stream << " --id " << sID << "\n";
            stream << " --fileIn " << sRaw << "\n";
            stream << " --events " << fileEvent.fileName() << "\n";
            stream << " --log " << bDoLogging << "\n";
            stream << " --logDir " << sLogDir << "\n";
            stream << " --cluster " << bDoCluster << "\n";
            stream << " --coilType " << sCoilType << "\n";
            stream << " --hpi " << bDoHPIFit << "\n";
            stream << " --movComp " << bDoHeadPosUpdate << "\n";
            stream << " --fastFit " << bDoFastFit << "\n";
            stream << " --maxMove " << dAllowedMovement * 1000 << "\n";
            stream << " --maxRot " << dAllowedRotation << "\n";
            stream << "\n";
            stream << "\n";
            stream << "Source Space:" << "\n";
            stream << "Number Spaces: " << pFwdSolution->src.size() << "\n";
            stream << "Number Sources: " << pFwdSolution->nsource << "\n";
            stream << "Number Sources Clustered: " << pClusteredFwd->nsource << "\n";
            stream << "\n";
            stream << "\n";
            stream << "HPI:" << "\n";
            stream << "Number coils: " << vecFreqs.size() << "\n";
            stream << "Frequencies: ";
            QVectorIterator<int> itFreqs(vecFreqs);
            while (itFreqs.hasNext()) {
                stream << itFreqs.next() << ",";
            }
            stream << "\n";
        }
        file.close();
    }

    //=============================================================================================================
    // actual pipeline

    QElapsedTimer timer;
    for(int i = 0; i < matEvents.rows(); i++) {
        from = matEvents(i,0) - 120;
        to = from + iQuantum;
        if (to > last) {
            to = last;
            qWarning() << "Block size < iQuantum " << iQuantum;
        }
        // Reading
        if(!pRawData->read_raw_segment(matData, matTimes, from, to)) {
            qCritical("error during read_raw_segment");
            return -1;
        }
        if(bDoHPIFit) {
            // coil ordering
            if(!bSorted) {
                while(!bSorted) {
                    qDebug() << "order coils";
                    bSorted = HPI.findOrder(matData,
                                            matSparseProjMult,
                                            transDevHeadFit,
                                            vecFreqs,
                                            fitResult.errorDistances,
                                            fitResult.GoF,
                                            fitResult.fittedCoils,
                                            pFiffInfo);
                }
                qDebug() << "Coil frequencies ordered: " << vecFreqs;
            }

            // HPI fit
            HPI.fitHPI(matData,
                       matSparseProjMult,
                       fitResult.devHeadTrans,
                       vecFreqs,
                       fitResult.errorDistances,
                       fitResult.GoF,
                       fitResult.fittedCoils,
                       pFiffInfo,
                       bDoDebug,
                       sHPIResourceDir);

            // get mean estimation error
            dMeanErrorDist = std::accumulate(fitResult.errorDistances.begin(), fitResult.errorDistances.end(), .0) / fitResult.errorDistances.size();
            HPIFit::storeHeadPosition((float)from/pFiffInfo->sfreq, fitResult.devHeadTrans.trans, matPosition, fitResult.GoF, fitResult.errorDistances);

            // update head position if good fit
            if(dMeanErrorDist < dErrorMax) {
//                HPIFit::storeHeadPosition((float)first/pFiffInfo->sfreq, fitResult.devHeadTrans.trans, matPosition, fitResult.GoF, fitResult.errorDistances);

                // check for big head movement
                dMovement = transDevHeadRef.translationTo(fitResult.devHeadTrans.trans);
                dRotation = transDevHeadRef.angleTo(fitResult.devHeadTrans.trans);
                if(dMovement > dAllowedMovement || dRotation > dAllowedRotation) {
                    matPosition(i,9) = 1;
                    fitResult.bIsLargeHeadMovement = true;
                    transDevHeadRef = fitResult.devHeadTrans;       // update reference head position
                } else {
                    fitResult.bIsLargeHeadMovement = false;
                }
                fitResult.bIsLargeHeadMovement = true;
            } else {
                qWarning() << "Bad Fit";
            }
            fitResult.bIsLargeHeadMovement = true;
            // update Forward solution
            if(bDoHeadPosUpdate) {
                if(fitResult.bIsLargeHeadMovement) {
                    timer.start();
                    transMegHeadOld = fitResult.devHeadTrans.toOld();
                    pComputeFwd->updateHeadPos(&transMegHeadOld);           // update
                    pFwdSolution->sol = pComputeFwd->sol;                   // store results
                    pFwdSolution->sol_grad = pComputeFwd->sol_grad;
                    vecTimeUpdate(i) = timer.elapsed();
                    // cluster
                    if(bDoCluster) {
                        pClusteredFwd = MNEForwardSolution::SPtr::create(pFwdSolution->cluster_forward_solution(*pAnnotationSet.data(), 20, defaultD, noiseCov));
                        forwardCompute = pClusteredFwd->pick_channels(lPickedChannels);
                    } else {
                        forwardCompute = pFwdSolution->pick_channels(lPickedChannels);
                    }
                    vecTimeUpdate(i) = timer.elapsed();
                }
            }
            if(bDoLogging) {
                QString sTimeStamp = QDateTime::currentDateTime().toString("yyMMdd_hhmmss");

                QString sSolName = sCurrentDir + "/" + sTimeStamp + "_" + sID + "-fwd.fif";
                pComputeFwd->storeFwd(sSolName);
            }
        }
    }

    // write hpi results
    if(bDoLogging) {
        QString sHPIFile = sID + "-pos.txt";
        QString sTimeFile = sID + "-updateTime.txt";
        IOUtils::write_eigen_matrix(matPosition, sCurrentDir + "/" + sHPIFile);
        IOUtils::write_eigen_matrix(vecTimeUpdate, sCurrentDir + "/" + sTimeFile);
    }

    qDebug() << "Done";
    return a.exec();;
}
