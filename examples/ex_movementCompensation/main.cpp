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

#include <rtprocessing/filter.h>
#include <rtprocessing/helpers/filterio.h>
#include <rtprocessing/rtcov.h>
#include <rtprocessing/rtave.h>
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

Qt3DCore::QTransform                        m_tAlignment;
QPointer<DISP3DLIB::BemTreeItem>            m_pBemHeadAvr;
QPointer<DISP3DLIB::DigitizerSetTreeItem>   m_pTrackedDigitizer;
Data3DTreeModel::SPtr                       m_p3DDataModel;
FIFFLIB::FiffEvokedSet                      m_evokedSet;
QStringList                                 m_lResponsibleTriggerTypes;

//=============================================================================================================
// functions
//=============================================================================================================

void alignFiducials(const QString& sFilePath)
{
    //Calculate the alignment of the fiducials
    MneMshDisplaySurfaceSet* pMneMshDisplaySurfaceSet = new MneMshDisplaySurfaceSet();
    MneMshDisplaySurfaceSet::add_bem_surface(pMneMshDisplaySurfaceSet,
                                             QCoreApplication::applicationDirPath() + "/resources/general/hpiAlignment/fsaverage-head.fif",
                                             FIFFV_BEM_SURF_ID_HEAD,
                                             "head",
                                             1,
                                             1);

    MneMshDisplaySurface* surface = pMneMshDisplaySurfaceSet->surfs[0];

    QFile t_fileDigData(sFilePath);
    FiffDigitizerData* t_digData = new FiffDigitizerData(t_fileDigData);

    QFile t_fileDigDataReference(QCoreApplication::applicationDirPath() + "/resources/general/hpiAlignment/fsaverage-fiducials.fif");

    float scales[3];
    QScopedPointer<FiffDigitizerData> t_digDataReference(new FiffDigitizerData(t_fileDigDataReference));
    MneSurfaceOrVolume::align_fiducials(t_digData,
                                        t_digDataReference.data(),
                                        surface,
                                        10,
                                        1,
                                        0,
                                        scales);

    QMatrix4x4 invMat;

    // use inverse transform
    for(int r = 0; r < 3; ++r) {
        for(int c = 0; c < 3; ++c) {
            // also apply scaling factor
            invMat(r,c) = t_digData->head_mri_t_adj->invrot(r,c) * scales[0];
        }
    }
    invMat(0,3) = t_digData->head_mri_t_adj->invmove(0);
    invMat(1,3) = t_digData->head_mri_t_adj->invmove(1);
    invMat(2,3) = t_digData->head_mri_t_adj->invmove(2);

    Qt3DCore::QTransform identity;
    m_tAlignment.setMatrix(invMat);

    // align and scale average head (now in head space)
    QList<QStandardItem*> itemList = m_pBemHeadAvr->findChildren(Data3DTreeModelItemTypes::BemSurfaceItem);
    for(int j = 0; j < itemList.size(); ++j) {
        if(BemSurfaceTreeItem* pBemItem = dynamic_cast<BemSurfaceTreeItem*>(itemList.at(j))) {
            pBemItem->setTransform(m_tAlignment);
        }
    }

    delete pMneMshDisplaySurfaceSet;
}

//=============================================================================================================

void update3DView(HpiFitResult fitResult) {
    // update 3D view
    m_p3DDataModel->addDigitizerData("Subject",
                                   "Fitted Digitizers",
                                   fitResult.fittedCoils.pickTypes(QList<int>()<<FIFFV_POINT_EEG));
    //Update fast scan / tracked digitizer
    QList<QStandardItem*> itemList = m_pTrackedDigitizer->findChildren(Data3DTreeModelItemTypes::DigitizerItem);
    for(int j = 0; j < itemList.size(); ++j) {
        if(DigitizerTreeItem* pDigItem = dynamic_cast<DigitizerTreeItem*>(itemList.at(j))) {
            // apply inverse to get from head to device space
            pDigItem->setTransform(fitResult.devHeadTrans, true);
        }
    }

    // Update average head
    itemList = m_pBemHeadAvr->findChildren(Data3DTreeModelItemTypes::BemSurfaceItem);
    for(int j = 0; j < itemList.size(); ++j) {
        if(BemSurfaceTreeItem* pBemItem = dynamic_cast<BemSurfaceTreeItem*>(itemList.at(j))) {
            pBemItem->setTransform(m_tAlignment);
            // apply inverse to get from head to device space
            pBemItem->applyTransform(fitResult.devHeadTrans, true);
        }
    }
}
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

void onNewEvokedSet(const FIFFLIB::FiffEvokedSet& evokedSet,
                    const QStringList& lResponsibleTriggerTypes) {

    m_lResponsibleTriggerTypes = lResponsibleTriggerTypes;
    m_evokedSet = evokedSet;
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
    parser.setApplicationDescription("Example name");
    parser.addHelpOption();

    QCommandLineOption parameterOption("parameter", "The first parameter description.");
    QCommandLineOption computeOption("compute", "Weather to compute or not.", "bool", "false");

    parser.addOption(computeOption);

    parser.process(a);

    //=============================================================================================================
    // Load data

    QFile t_fileRaw(QCoreApplication::applicationDirPath() + "/MNE-sample-data/simulate/sim-chpi-move-aud-raw.fif");
    QFile t_fileMriName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/MEG/sample/all-trans.fif");
    QFile t_fileBemName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem.fif");
    QFile t_fileSrcName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/bem/sample-oct-6-src.fif");
    QFile t_fileMeasName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/MEG/sample/sample_audvis_raw.fif");

    QString sFilterPath(QCoreApplication::applicationDirPath() + "/MNE-sample-data/filterBPF.txt");
    QString sAtlasDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/label");
    QString sSurfaceDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/surf");

    FiffRawData rawData(t_fileRaw);
    FiffRawData rawMeas(t_fileMeasName);

    QSharedPointer<FiffInfo> pFiffInfo = QSharedPointer<FiffInfo>(new FiffInfo(rawData.info));

    QStringList exclude;
    exclude << rawData.info.bads << rawData.info.ch_names.filter("EOG") << "MEG2641" << "MEG2443" << "EEG053";

//    // only compute on grad and meg
//    RowVectorXi vecPicks = rawData.info.pick_types(QString("grad"),false,true,QStringList(),exclude);
//    FiffInfo::SPtr pFiffInfoCompute = QSharedPointer<FiffInfo>(new FiffInfo(pFiffInfo->pick_info(vecPicks)));
//    QStringList pickedChannels = pFiffInfoCompute->ch_names;

    double dSFreq = pFiffInfo->sfreq;

    AnnotationSet::SPtr pAnnotationSet = AnnotationSet::SPtr(new AnnotationSet(sAtlasDir+"/lh.aparc.a2009s.annot", sAtlasDir+"/rh.aparc.a2009s.annot"));
    FiffCoordTrans mriHeadTrans(t_fileMriName); // mri <-> head transformation matrix

    //=============================================================================================================
    // Setup GUI and load average head

    AbstractView::SPtr p3DAbstractView = AbstractView::SPtr(new AbstractView());
    m_p3DDataModel = p3DAbstractView->getTreeModel();

    QFile t_fileVVSensorSurfaceBEM(QCoreApplication::applicationDirPath() + "/resources/general/sensorSurfaces/306m.fif");
    MNEBem t_sensorVVSurfaceBEM(t_fileVVSensorSurfaceBEM);
    m_p3DDataModel->addMegSensorInfo("Device", "VectorView", QList<FiffChInfo>(), t_sensorVVSurfaceBEM);

    QFile t_fileHeadAvr(QCoreApplication::applicationDirPath() + "/resources/general/hpiAlignment/fsaverage-head.fif");;
    MNEBem t_BemHeadAvr(t_fileHeadAvr);
    m_pBemHeadAvr = m_p3DDataModel->addBemData("Subject", "Average head", t_BemHeadAvr);


    FiffDigPointSet digSet(t_fileRaw);
    FiffDigPointSet digSetWithoutAdditional = digSet.pickTypes(QList<int>()<<FIFFV_POINT_HPI<<FIFFV_POINT_CARDINAL<<FIFFV_POINT_EEG);
    m_pTrackedDigitizer = m_p3DDataModel->addDigitizerData("Subject",
                                                                                                 "Tracked Digitizers",
                                                                                                 digSetWithoutAdditional);
    alignFiducials(t_fileRaw.fileName());

    p3DAbstractView->show();

    //=============================================================================================================
    // Setup customisable values
    // Data Stream

    fiff_int_t iQuantum = 200;              // Buffer size

    // HPI Fitting
    double dErrorMax = 0.10;                // maximum allowed estimation error
    double dAllowedMovement = 0.003;        // maximum allowed movement
    double dAllowedRotation = 5;            // maximum allowed rotation
    bool bDoFastFit = true;                 // fast fit or advanced linear model

    //=============================================================================================================
    // setup data stream

    MatrixXd matData, matTimes;
    MatrixXd matPosition;                   // matPosition matrix to save quaternions etc.

    fiff_int_t from, to;
    fiff_int_t first = rawData.first_samp;
    fiff_int_t last = rawData.last_samp;

    int iN = ceil((last-first)/iQuantum);   // number of data segments
    MatrixXd vecTime = RowVectorXd::LinSpaced(iN, 0, iN-1)*iQuantum;

    Eigen::SparseMatrix<double> matSparseProjMult = updateProjectors(*(pFiffInfo.data()));

    //=============================================================================================================
    // setup Covariance

    int iEstimationSamples = 2000;

    FiffCov fiffCov;
    FiffCov fiffComputedCov;

    RTPROCESSINGLIB::RtCov rtCov(pFiffInfo);

    //=============================================================================================================
    // setup averaging

    QMap<QString,int>       mapStimChsIndexNames;
    QMap<QString,double>    mapThresholdsFirst;
    QMap<QString,int>       mapThresholdsSecond;
    QMap<QString,double>    mapThresholds;

    for(qint32 i = 0; i < pFiffInfo->chs.size(); ++i) {
        if(pFiffInfo->chs[i].kind == FIFFV_STIM_CH) {
            mapStimChsIndexNames.insert(pFiffInfo->chs[i].ch_name,i);
        }
    }

    QStringList lResponsibleTriggerTypes;
    lResponsibleTriggerTypes << "3";

    bool bDoArtifactThresholdReduction = true;
    int iPreStimSamples = 0*dSFreq;
    int iPostStimSamples = ((float)250/1000)*dSFreq;
    int iBaselineFromSamples = ((float)0/1000)*dSFreq;
    int iBaselineToSamples = ((float)210/1000)*dSFreq;
    int iNumAverages = 10;
    int iCurrentStimChIdx = mapStimChsIndexNames.value("STI001");

    QSharedPointer<RTPROCESSINGLIB::RtAve> pRtAve;
    pRtAve = RtAve::SPtr::create(iNumAverages,
                                 iPreStimSamples,
                                 iPostStimSamples,
                                 iBaselineFromSamples,
                                 iBaselineToSamples,
                                 iCurrentStimChIdx,
                                 pFiffInfo);

    pRtAve->setBaselineFrom(iBaselineFromSamples, iBaselineFromSamples/dSFreq);
    pRtAve->setBaselineTo(iBaselineToSamples, iBaselineToSamples/dSFreq);
    pRtAve->setBaselineActive(true);

    if(bDoArtifactThresholdReduction) {
        mapThresholds["Active"] = 1.0;
    } else {
        mapThresholds["Active"] = 0.0;
    }

    mapThresholdsFirst["grad"] = 1.0;
    mapThresholdsFirst["mag"] = 1.0;
    mapThresholdsFirst["eeg"] = 1.0;
    mapThresholdsFirst["ecg"] = 1.0;
    mapThresholdsFirst["emg"] = 1.0;
    mapThresholdsFirst["eog"] = 1.0;

    mapThresholdsSecond["grad"] = -1;
    mapThresholdsSecond["mag"] = -1;
    mapThresholdsSecond["eeg"] = -1;
    mapThresholdsSecond["ecg"] = -1;
    mapThresholdsSecond["emg"] = -1;
    mapThresholdsSecond["eog"] = -1;

    pRtAve->setArtifactReduction(mapThresholds);

    FIFFLIB::FiffEvokedSet evokedSet;

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
    FiffCoordTrans transDevHeadRef = pFiffInfo->dev_head_t;     // reference of last fit
    FiffCoordTrans transDevHeadFit = pFiffInfo->dev_head_t;
    // create HPI fit object
    HPIFit HPI = HPIFit(pFiffInfo,bDoFastFit);

    //=============================================================================================================
    // setup Forward solution

    QFile t_fileAtlasDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/label");

    ComputeFwdSettings::SPtr pFwdSettings = ComputeFwdSettings::SPtr(new ComputeFwdSettings);
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
    pFwdSettings->pFiffInfo = pFiffInfo;

    FiffCoordTransOld transMegHeadOld = fitResult.devHeadTrans.toOld();
    QFile t_fSolution(pFwdSettings->solname);

    MNEForwardSolution::SPtr pFwdSolution;
    MNEForwardSolution::SPtr pClusteredFwd;

    // init Forward solution
    ComputeFwd::SPtr pComputeFwd = ComputeFwd::SPtr(new ComputeFwd(pFwdSettings));

    // compute and store first forward solution
    pComputeFwd->calculateFwd();
    pComputeFwd->storeFwd();

    Eigen::MatrixXd defaultD;       // default cluster operator

    pFwdSolution = MNEForwardSolution::SPtr(new MNEForwardSolution(t_fSolution, false, true));
    pClusteredFwd = MNEForwardSolution::SPtr(new MNEForwardSolution(pFwdSolution->cluster_forward_solution(*pAnnotationSet.data(), 200,defaultD,fiffComputedCov)));

    // ToDo Plot in head space
    QList<SourceSpaceTreeItem*> pSourceSpaceItem = m_p3DDataModel->addForwardSolution("Subject", "ClusteredForwardSolution", *pClusteredFwd);
    QList<SourceSpaceTreeItem*> pClusteredSourceSpaceItem = m_p3DDataModel->addForwardSolution("Subject", "ForwardSolution", *pFwdSolution);


    //=============================================================================================================
    // setup MNE source estimate

    QString sAvrType = "3";
    QString sMethod = "dSPM";
    SurfaceSet::SPtr pSurfaceSet = SurfaceSet::SPtr(new SurfaceSet(sSurfaceDir+"/lh.orig", sSurfaceDir+"/rh.orig"));
    RtInvOp::SPtr pRtInvOp = RtInvOp::SPtr(new RtInvOp(pFiffInfo, pFwdSolution));
    MNELIB::MNEInverseOperator invOp;
    FiffEvoked currentEvoked;

    qint32 skip_count = 0;
    FiffEvoked evoked;
    MatrixXd matDataResized;
    qint32 j;
    int iTimePointSps = 0.187*dSFreq;
    int iNumberChannels = 0;
    int iDownSample = 1;
    float tstep = 1 / dSFreq;
    float lambda2 = 1.0f / pow(1.0f, 2); //ToDo estimate lambda using covariance
    MNESourceEstimate sourceEstimate;
    bool bUpdateMinimumNorm = false;
    QSharedPointer<INVERSELIB::MinimumNorm> pMinimumNorm;
    QStringList lChNamesFiffInfo;
    QStringList lChNamesInvOp;

    //=============================================================================================================
    // setup Filter
    double dFromFreq = 1.0;
    double dToFreq = 40.0;
    double dTransWidth = 0.1;
    double dBw = dToFreq-dFromFreq;
    double dCenter = dFromFreq+dBw/2.0;
    double nyquistFrequency = dSFreq/2.0;
    int iFilterTabs = 128;
    FilterKernel::DesignMethod dMethod = FilterKernel::Cosine;
    QScopedPointer<RTPROCESSINGLIB::Filter> pRtFilter(new RTPROCESSINGLIB::Filter());
    RTPROCESSINGLIB::FilterKernel filterKernel = FilterKernel("Designed Filter",
                                                              FilterKernel::BPF,
                                                              iFilterTabs,
                                                              (double)dCenter/nyquistFrequency,
                                                              (double)dBw/nyquistFrequency,
                                                              (double)dTransWidth/nyquistFrequency,
                                                              (double)dSFreq,
                                                              dMethod);

    FilterIO::readFilter(sFilterPath, filterKernel);

    Eigen::RowVectorXi lFilterChannelList;
    for(int i = 0; i < pFiffInfo->chs.size(); ++i) {
        if((pFiffInfo->chs.at(i).kind == FIFFV_MEG_CH || pFiffInfo->chs.at(i).kind == FIFFV_EEG_CH ||
            pFiffInfo->chs.at(i).kind == FIFFV_EOG_CH || pFiffInfo->chs.at(i).kind == FIFFV_ECG_CH ||
            pFiffInfo->chs.at(i).kind == FIFFV_EMG_CH)/* && !pFiffInfo->bads.contains(pFiffInfo->chs.at(i).ch_name)*/) {

            lFilterChannelList.conservativeResize(lFilterChannelList.cols() + 1);
            lFilterChannelList[lFilterChannelList.cols()-1] = i;
        }
    }
    //=============================================================================================================
    // prepare writing
    RowVectorXd cals;
    QFile t_fileOut(QCoreApplication::applicationDirPath() + "/MNE-sample-data/test_filter_raw.fif");
    FiffStream::SPtr outfid = FiffStream::start_writing_raw(t_fileOut, rawData.info,cals);
    MatrixXd matResultFiltered(lFilterChannelList.size(),last-first);
    bool first_buffer = true;

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

        // get error
        dMeanErrorDist = std::accumulate(fitResult.errorDistances.begin(), fitResult.errorDistances.end(), .0) / fitResult.errorDistances.size();
        // update head position
        if(dMeanErrorDist < dErrorMax) {
            HPIFit::storeHeadPosition(first/pFiffInfo->sfreq, fitResult.devHeadTrans.trans, matPosition, fitResult.GoF, fitResult.errorDistances);

            // update 3D View
            update3DView(fitResult);

            // check displacement
            dMovement = transDevHeadRef.translationTo(fitResult.devHeadTrans.trans);
            dRotation = transDevHeadRef.angleTo(fitResult.devHeadTrans.trans);
            if(dMovement > dAllowedMovement || dRotation > dAllowedRotation) {
                matPosition(i,9) = 1;
                fitResult.bIsLargeHeadMovement = true;
                transDevHeadRef = fitResult.devHeadTrans;       // update reference head position
            } else {
                fitResult.bIsLargeHeadMovement = false;
            }
        }

        // update Forward solution:
//        if(fitResult.bIsLargeHeadMovement) {
//            transMegHeadOld = fitResult.devHeadTrans.toOld();
//            pComputeFwd->updateHeadPos(&transMegHeadOld);
//            pFwdSolution->sol = pComputeFwd->sol;
//            pFwdSolution->sol_grad = pComputeFwd->sol_grad;
//            pClusteredFwd = MNEForwardSolution::SPtr(new MNEForwardSolution(pFwdSolution->cluster_forward_solution(*pAnnotationSet.data(), 200, defaultD, fiffComputedCov)));
//            pRtInvOp->setFwdSolution(pFwdSolution);
//            pRtInvOp->append(fiffComputedCov);
        //        }

        // Filtering
        matData = matSparseProjMult * matData;
        QList<FilterKernel> list;
        list << filterKernel;
        matData = pRtFilter->filterData(matData,
                                        list,
                                        lFilterChannelList);

        // Covariance
        fiffCov = rtCov.estimateCovariance(matData, iEstimationSamples);
        if(!fiffCov.names.isEmpty()) {
            fiffComputedCov = fiffCov;
            qDebug() << "adg";
            pRtInvOp->append(fiffComputedCov);
            qDebug() << "Covariance updated";
        }

        // averaging
        // Init the stim channels
        for(qint32 i = 0; i < pFiffInfo->chs.size(); ++i) {
            if(pFiffInfo->chs[i].kind == FIFFV_STIM_CH) {
                mapStimChsIndexNames.insert(pFiffInfo->chs[i].ch_name,i);
            }
        }
        pRtAve->append(matData);
        qDebug() << "oha" ;
        QObject::connect(pRtAve.data(), &RtAve::evokedStim, onNewEvokedSet);
        evokedSet = m_evokedSet;
        lResponsibleTriggerTypes = m_lResponsibleTriggerTypes;

        if(evokedSet.evoked.at(0).nave > 0) {
            currentEvoked = evokedSet.evoked.at(0);
            DISPLIB::ImageSc imagesc(currentEvoked.data);
            imagesc.setTitle("Data ImageSc");
            imagesc.setXLabel("X Axes");
            imagesc.setYLabel("Y Axes");
            imagesc.setColorMap("Hot");//imagesc.setColorMap("Jet");//imagesc.setColorMap("RedBlue");//imagesc.setColorMap("Bone");//imagesc.setColorMap("Jet");//imagesc.setColorMap("Hot");
            imagesc.show();
        }


        // source estimate
        FiffEvokedSet::SPtr pFiffEvokedSet = FiffEvokedSet::SPtr(new FiffEvokedSet(evokedSet));

//        iNumberChannels = invOp.noise_cov->names.size();
//        lChNamesFiffInfo = pickedChannels;
//        lChNamesInvOp = invOp.noise_cov->names;

//        if(bUpdateMinimumNorm) {
//            qDebug() << "hier?";
//            pMinimumNorm = MinimumNorm::SPtr(new MinimumNorm(invOp, lambda2, sMethod));
//            bUpdateMinimumNorm = false;
//            // Set up the inverse according to the parameters.
//            // Use 1 nave here because in case of evoked data as input the minimum norm will always be updated when the source estimate is calculated (see run method).
//            pMinimumNorm->doInverseSetup(1,true);
//        }

//        for(int i = 0; i < pFiffEvokedSet->evoked.size(); ++i) {
//            qDebug() << "oder hier?";
//            if(pFiffEvokedSet->evoked.at(i).comment == sAvrType) {
//                // Get the current evoked data
//                qDebug() << "a";
//                currentEvoked = pFiffEvokedSet->evoked.at(i).pick_channels(pickedChannels);
//                qDebug() << "a2";
//                sourceEstimate = pMinimumNorm->calculateInverse(currentEvoked);

//                if(((skip_count % iDownSample) == 0)) {
//                    qDebug() << "b";
//                    sourceEstimate = pMinimumNorm->calculateInverse(currentEvoked);
//                    qDebug() << "c";
//                    if(!sourceEstimate.isEmpty()) {
//                        if(iTimePointSps < sourceEstimate.data.cols() && iTimePointSps >= 0) {
//                            qDebug() << "d";
//                            sourceEstimate = sourceEstimate.reduce(iTimePointSps,1);
//                            std::cout << sourceEstimate.data << std::endl;
//                        }
//                    }
//                }

//                ++skip_count;

//            }
//        }


        // Write Data
        if(first_buffer) {
            if(first > 0) {
                outfid->write_int(FIFF_FIRST_SAMPLE, &first);
            }
            first_buffer = false;
        }
        outfid->write_raw_buffer(matData,cals);



    }
    outfid->finish_writing_raw();
    qDebug() << "Done";
    return a.exec();
}
