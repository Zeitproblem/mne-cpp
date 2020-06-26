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
QStringList                                 m_qListCovChNames;          /**< Covariance channel names. */
QStringList                                 m_qListPickChannels;        /**< Channels to pick. */
QSharedPointer<FIFFLIB::FiffInfo>           m_pFiffInfoInput;           /**< Fiff information of the evoked. */
QSharedPointer<FIFFLIB::FiffInfoBase>       m_pFiffInfoForward;         /**< Fiff information of the forward solution. */
QSharedPointer<FIFFLIB::FiffInfo>           m_pFiffInfo;                /**< Fiff information. */

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

bool calcFiffInfo()
{

    if(m_qListCovChNames.size() > 0 && m_pFiffInfoInput && m_pFiffInfoForward) {
        qDebug() << "[RtcMne::calcFiffInfoFiff] Infos available";

        //        qDebug() << "RtcMne::calcFiffInfo - m_qListCovChNames" << m_qListCovChNames;
        //        qDebug() << "RtcMne::calcFiffInfo - m_pFiffInfoForward->ch_names" << m_pFiffInfoForward->ch_names;
        //        qDebug() << "RtcMne::calcFiffInfo - m_pFiffInfoInput->ch_names" << m_pFiffInfoInput->ch_names;

        // Align channel names of the forward solution to the incoming averaged (currently acquired) data
        // Find out whether the forward solution depends on only MEG, EEG or both MEG and EEG channels
        QStringList forwardChannelsTypes;
        m_pFiffInfoForward->ch_names.clear();
        int counter = 0;

        for(qint32 x = 0; x < m_pFiffInfoForward->chs.size(); ++x) {
            if(forwardChannelsTypes.contains("MEG") && forwardChannelsTypes.contains("EEG"))
                break;

            if(m_pFiffInfoForward->chs[x].kind == FIFFV_MEG_CH && !forwardChannelsTypes.contains("MEG"))
                forwardChannelsTypes<<"MEG";

            if(m_pFiffInfoForward->chs[x].kind == FIFFV_EEG_CH && !forwardChannelsTypes.contains("EEG"))
                forwardChannelsTypes<<"EEG";
        }

        //If only MEG channels are used
        if(forwardChannelsTypes.contains("MEG") && !forwardChannelsTypes.contains("EEG")) {
            for(qint32 x = 0; x < m_pFiffInfoInput->chs.size(); ++x)
            {
                if(m_pFiffInfoInput->chs[x].kind == FIFFV_MEG_CH) {
                    m_pFiffInfoForward->chs[counter].ch_name = m_pFiffInfoInput->chs[x].ch_name;
                    m_pFiffInfoForward->ch_names << m_pFiffInfoInput->chs[x].ch_name;
                    counter++;
                }
            }
        }

        //If only EEG channels are used
        if(!forwardChannelsTypes.contains("MEG") && forwardChannelsTypes.contains("EEG")) {
            for(qint32 x = 0; x < m_pFiffInfoInput->chs.size(); ++x)
            {
                if(m_pFiffInfoInput->chs[x].kind == FIFFV_EEG_CH) {
                    m_pFiffInfoForward->chs[counter].ch_name = m_pFiffInfoInput->chs[x].ch_name;
                    m_pFiffInfoForward->ch_names << m_pFiffInfoInput->chs[x].ch_name;
                    counter++;
                }
            }
        }

        //If both MEG and EEG channels are used
        if(forwardChannelsTypes.contains("MEG") && forwardChannelsTypes.contains("EEG")) {
            //qDebug()<<"RtcMne::calcFiffInfo - MEG EEG fwd solution";
            for(qint32 x = 0; x < m_pFiffInfoInput->chs.size(); ++x)
            {
                if(m_pFiffInfoInput->chs[x].kind == FIFFV_MEG_CH || m_pFiffInfoInput->chs[x].kind == FIFFV_EEG_CH) {
                    m_pFiffInfoForward->chs[counter].ch_name = m_pFiffInfoInput->chs[x].ch_name;
                    m_pFiffInfoForward->ch_names << m_pFiffInfoInput->chs[x].ch_name;
                    counter++;
                }
            }
        }

        //Pick only channels which are present in all data structures (covariance, evoked and forward)
        QStringList tmp_pick_ch_names;
        foreach (const QString &ch, m_pFiffInfoForward->ch_names)
        {
            if(m_pFiffInfoInput->ch_names.contains(ch))
                tmp_pick_ch_names << ch;
        }
        m_qListPickChannels.clear();

        foreach (const QString &ch, tmp_pick_ch_names)
        {
            if(m_qListCovChNames.contains(ch))
                m_qListPickChannels << ch;
        }
        RowVectorXi sel = m_pFiffInfoInput->pick_channels(m_qListPickChannels);

        //qDebug() << "RtcMne::calcFiffInfo - m_qListPickChannels.size()" << m_qListPickChannels.size();
        //qDebug() << "RtcMne::calcFiffInfo - m_qListPickChannels" << m_qListPickChannels;

        m_pFiffInfo = QSharedPointer<FiffInfo>(new FiffInfo(m_pFiffInfoInput->pick_info(sel)));

        // qDebug() << "RtcMne::calcFiffInfo - m_pFiffInfo" << m_pFiffInfo->ch_names;

        return true;
    }

    return false;
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
    parser.setApplicationDescription("Evaluation for movement compensation");
    parser.addHelpOption();

    QCommandLineOption writeOption("write", "Write the filtered data to .fif.", "bool", "false");
    QCommandLineOption moveCompOption("movComp", "Perform head position updates", "bool", "false");
    QCommandLineOption hpiOption("hpi", "Is HPI fitting acquired?", "bool", "true");
    QCommandLineOption fastFitOption("fastFit", "Do fast HPI fits.", "bool", "true");
    QCommandLineOption maxMoveOption("maxMove", "Maximum head movement in mm.", "value", "3");
    QCommandLineOption maxRotOption("maxRot", "Maximum head rotation in degree.", "value", "5");
    QCommandLineOption coilTypeOption("coilType", "The coil <type> (for sensor level usage only), 'eeg', 'grad' or 'mag'.", "type", "grad");
    QCommandLineOption clustOption("cluster", "Do clustering of source space (for source level usage only).", "doClust", "true");
    QCommandLineOption sourceLocMethodOption("sourceLocMethod", "Inverse estimation <method> (for source level usage only), i.e., 'MNE', 'dSPM' or 'sLORETA'.", "method", "dSPM");
    QCommandLineOption rawFileOption("fileIn", "The input file <in>.", "in", QCoreApplication::applicationDirPath() + "/MNE-sample-data/simulate/sim-chpi-move-aud-raw.fif");
    QCommandLineOption logOption("log", "Log the computation and store results to file.", "bool", "true");
    QCommandLineOption logDirOption("logDir", "The directory to log the results.", "string", QCoreApplication::applicationDirPath() + "/MNE-sample-data/chpi/evaluation");
    QCommandLineOption idOption("id", "The id for the measurement", "id", "id");

    parser.addOption(writeOption);
    parser.addOption(moveCompOption);
    parser.addOption(hpiOption);
    parser.addOption(fastFitOption);
    parser.addOption(maxMoveOption);
    parser.addOption(maxRotOption);
    parser.addOption(coilTypeOption);
    parser.addOption(clustOption);
    parser.addOption(sourceLocMethodOption);
    parser.addOption(rawFileOption);
    parser.addOption(logOption);
    parser.addOption(logDirOption);
    parser.addOption(idOption);

    parser.process(a);

    //=============================================================================================================
    // Setup customisable values via command line

    // Data Stream
    QString sRaw = parser.value(rawFileOption);

    bool bWriteFilteredData = false;        // write filtered data to .fif

    if(parser.value(writeOption) == "false" || parser.value(writeOption) == "0") {
        bWriteFilteredData = false;
    } else if(parser.value(writeOption) == "true" || parser.value(writeOption) == "1") {
        bWriteFilteredData = true;
    }

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

    // Forward Solution#
    bool bDoHeadPosUpdate = false;
    bool bDoCluster = true;                 // not implemented, always cluster

    if(parser.value(moveCompOption) == "false" || parser.value(moveCompOption) == "0") {
        bDoHeadPosUpdate = false;
    } else if(parser.value(moveCompOption) == "true" || parser.value(moveCompOption) == "1") {
        bDoHeadPosUpdate = true;
    }

    // Noise Covariance
    int iEstimationSamples = 2000;

    // Source estimation
    QString sMethod = parser.value(sourceLocMethodOption);              // source estimation method "MNE" "dSPM" "sLORETA"

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
    // Load data

    QFile t_fileMriName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/MEG/sample/all-trans.fif");
    QFile t_fileBemName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem.fif");
    QFile t_fileSrcName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/bem/sample-oct-6-src.fif");
    QFile t_fileMeasName(QCoreApplication::applicationDirPath() + "/MNE-sample-data/MEG/sample/sample_audvis_raw.fif");
    QFile t_fileCov(QCoreApplication::applicationDirPath() + "/MNE-sample-data/MEG/sample/sample_audvis-cov.fif");

    QString sFilterPath(QCoreApplication::applicationDirPath() + "/MNE-sample-data/filterBPF.txt");
    QString sAtlasDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/label");
    QString sSurfaceDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/surf");


    QFile t_fileRaw(sRaw);

    FiffRawData rawData(t_fileRaw);
    FiffRawData rawMeas(t_fileMeasName);

    // Setup compensators and projectors so they get applied while reading
    bool keep_comp = false;
    fiff_int_t dest_comp = 0;
//    MNE::setup_compensators(rawData,
//                            dest_comp,
//                            keep_comp);

    QSharedPointer<FiffInfo> pFiffInfo = QSharedPointer<FiffInfo>(new FiffInfo(rawData.info));

    QStringList exclude;
    exclude << rawData.info.bads << rawData.info.ch_names.filter("EOG") << "MEG2641" << "MEG2443" << "EEG053";

    RowVectorXi vecPicks;
    // pick sensor types
    if(sCoilType.contains("grad", Qt::CaseInsensitive)) {
        vecPicks = rawData.info.pick_types(QString("grad"),false,true,QStringList(),exclude);
    } else if (sCoilType.contains("mag", Qt::CaseInsensitive)) {
        vecPicks = rawData.info.pick_types(QString("mag"),false,true,QStringList(),exclude);
    } else if (sCoilType.contains("eeg", Qt::CaseInsensitive)) {
        vecPicks = rawData.info.pick_types(QString("all"),true,true,QStringList(),exclude);
    } else {
        vecPicks = rawData.info.pick_types(QString("all"),false,true,QStringList(),exclude);
    }

    FiffInfo::SPtr pFiffInfoCompute = QSharedPointer<FiffInfo>(new FiffInfo(pFiffInfo->pick_info(vecPicks)));
    QStringList pickedChannels = pFiffInfoCompute->ch_names;

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
    m_pTrackedDigitizer = m_p3DDataModel->addDigitizerData("Subject","Tracked Digitizers",digSetWithoutAdditional);
    alignFiducials(t_fileRaw.fileName());

    SurfaceSet::SPtr pSurfaceSet = SurfaceSet::SPtr(new SurfaceSet(sSurfaceDir+"/lh.orig", sSurfaceDir+"/rh.orig"));
    m_p3DDataModel->addSurfaceSet("Subject", "MRI", *pSurfaceSet.data(), *pAnnotationSet.data());

    // buterfly plot
//    DISPLIB::ButterflyView::SPtr m_pButterflyView = DISPLIB::ButterflyView::SPtr(new DISPLIB::ButterflyView());
//    QSharedPointer<DISPLIB::EvokedSetModel> m_pEvokedModel;
//    QSharedPointer<DISPLIB::ChannelInfoModel> m_pChannelInfoModel = DISPLIB::ChannelInfoModel::SPtr::create(pFiffInfo);
//    m_pEvokedModel->setEvokedSet(pFiffEvokedSet);
//    m_pButterflyView->setEvokedSetModel(m_pEvokedModel);
//    m_pButterflyView->setChannelInfoModel(m_pChannelInfoModel);
//    m_pButterflyView->dataUpdate();
//    m_pButterflyView->updateView();

    //=============================================================================================================
    // setup data stream

    MatrixXd matData, matTimes;
    MatrixXd matPosition;                   // matPosition matrix to save quaternions etc.

    fiff_int_t iQuantum = 200;              // Buffer size
    fiff_int_t from, to;
    fiff_int_t first = rawData.first_samp;
    fiff_int_t last = rawData.last_samp;

    int iN = ceil((last-first)/iQuantum);   // number of data segments
    MatrixXd vecTime = RowVectorXd::LinSpaced(iN, 0, iN-1)*iQuantum;

    Eigen::SparseMatrix<double> matSparseProjMult = updateProjectors(*(pFiffInfo.data()));

    //=============================================================================================================
    // setup Covariance

//    FiffCov fiffCov;
//    FiffCov fiffComputedCov;

//    RTPROCESSINGLIB::RtCov rtCov(pFiffInfo);

    FiffCov noiseCov(t_fileCov);
    noiseCov = noiseCov.regularize(*pFiffInfoCompute.data(), 0.05, 0.05, 0.1, true);

    //=============================================================================================================
    // setup averaging

    QMap<QString,int>       mapStimChsIndexNames;
    QMap<QString,double>    mapThresholds;

    for(qint32 i = 0; i < pFiffInfo->chs.size(); ++i) {
        if(pFiffInfo->chs[i].kind == FIFFV_STIM_CH) {
            mapStimChsIndexNames.insert(pFiffInfo->chs[i].ch_name,i);
        }
    }

    QStringList lResponsibleTriggerTypes;
    lResponsibleTriggerTypes << "3";

    // define source time series
    bool bDoArtifactThresholdReduction = false;
    int iPreStimSamples = 0;
    int iPostStimSamples = 125;
    int iBaselineFromSamples = 0;
    int iBaselineToSamples = 100;
    int iNumAverages = 10;
    QString sStimChan = "STI001";
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
    HPIFit HPI = HPIFit(pFiffInfo,bDoFastFit);

    //=============================================================================================================
    // setup Forward solution

    QFile t_fileAtlasDir(QCoreApplication::applicationDirPath() + "/MNE-sample-data/subjects/sample/label");
    VectorXd vecTimeUpdate = vecTime;       // store time elapsed during head position update
    if(!bDoHeadPosUpdate) {
        vecTimeUpdate.fill(0);
    }
    ComputeFwdSettings::SPtr pFwdSettings = ComputeFwdSettings::SPtr(new ComputeFwdSettings);
    pFwdSettings->solname = QCoreApplication::applicationDirPath() + "/MNE-sample-data/your-solution-fwd.fif";
    pFwdSettings->mriname = t_fileMriName.fileName();
    pFwdSettings->bemname = t_fileBemName.fileName();
    pFwdSettings->srcname = t_fileSrcName.fileName();
    pFwdSettings->measname = t_fileMeasName.fileName();
    pFwdSettings->transname.clear();
    pFwdSettings->eeg_model_name = "Default";
    pFwdSettings->include_meg = true;
    pFwdSettings->include_eeg = false;
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

    pFwdSolution = MNEForwardSolution::SPtr(new MNEForwardSolution(t_fSolution));
    pClusteredFwd = MNEForwardSolution::SPtr(new MNEForwardSolution(pFwdSolution->cluster_forward_solution(*pAnnotationSet.data(), 20,defaultD,noiseCov/*fiffComputedCov*/)));
    m_pFiffInfoForward = FIFFLIB::FiffInfoBase::SPtr(new FIFFLIB::FiffInfoBase(pClusteredFwd->info));

    MNEForwardSolution forwardMeg = pClusteredFwd->pick_types(true, false);

    // ToDo Plot in head space
    QList<SourceSpaceTreeItem*> pSourceSpaceItem = m_p3DDataModel->addForwardSolution("Subject", "ClusteredForwardSolution", *pClusteredFwd);
    QList<SourceSpaceTreeItem*> pClusteredSourceSpaceItem = m_p3DDataModel->addForwardSolution("Subject", "ForwardSolution", *pFwdSolution);

    //=============================================================================================================
    // setup MNE source estimate

    QString sAvrType = "3";
    RtInvOp::SPtr pRtInvOp = RtInvOp::SPtr(new RtInvOp(pFiffInfoCompute, pFwdSolution));
    MNELIB::MNEInverseOperator invOp = MNEInverseOperator(*pFiffInfoCompute.data(),                   // create new inverse operator
                                                          forwardMeg,
                                                          noiseCov,
                                                          0.2f,
                                                          0.8f);;
    FiffEvoked currentEvoked;

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
    // Logging

    qInfo() << "-------------------------------------------------------------------------------------------------";
    qInfo() << "File: " << sRaw;
    qInfo() << "Write filtered data to file: " << bWriteFilteredData;
    qInfo() << "Coil Type: " << sCoilType;
    qInfo() << "Fast HPI Fit: " << bDoFastFit;
    qInfo() << "Movement compensation: " << bDoHeadPosUpdate;
    qInfo() << "Maximum rotation: " << dAllowedRotation;
    qInfo() << "Maximum displacement" << dAllowedMovement;
    qInfo() << "Clustered Sources" << forwardMeg.nsource;
    qInfo() << "-------------------------------------------------------------------------------------------------";

    QString sCurrentDir;
    if(bDoLogging) {

        QString sTimeStamp = QDateTime::currentDateTime().toString("yyMMdd_hhmmss");

        if(!QDir(sLogDir).exists()) {
            QDir().mkdir(sLogDir);
        }

        sCurrentDir = sLogDir + "/" + sTimeStamp + "_" + sID;
        QDir().mkdir(sCurrentDir);
        QString sLogFile = sID + "_LogFile.txt";
        QFile file(sCurrentDir + "/" + sLogFile);
        qDebug() << file.fileName();
        qDebug() << sLogFile;
        qDebug() << sCurrentDir;
        if(file.open(QIODevice::WriteOnly|QIODevice::Truncate)) {
            QTextStream stream(&file);
            stream << "ex_movementCompensation" << "\n";
            stream << " --id " << sID << "\n";
            stream << " --fileIn " << sRaw << "\n";
            stream << " --log " << bDoLogging << "\n";
            stream << " --logDir " << sLogDir << "\n";
            stream << " --write " << bWriteFilteredData << "\n";
            stream << " --sourceLocMethod " << sMethod << "\n";
            stream << " --cluster " << bDoCluster << "\n";
            stream << " --coilType " << sCoilType << "\n";
            stream << " --hpi " << bDoHPIFit << "\n";
            stream << " --movComp " << bDoHeadPosUpdate << "\n";
            stream << " --fastFit " << bDoFastFit << "\n";
            stream << " --maxMove " << dAllowedMovement * 1000 << "\n";
            stream << " --maxRot " << dAllowedRotation << "\n";
            stream << "\n";
            stream << "\n";
            stream << "Filter:" << "\n";
            stream << "Type: " << "BPF" << "\n";
            stream << "Method: " << "Cosine" << "\n";
            stream << "From: " << dFromFreq << "\n";
            stream << "To: " << dToFreq << "\n";
            stream << "BandWith: " << dBw << "\n";
            stream << "Transition band: " << dTransWidth << "\n";
            stream << "Filter Tabs: " << iFilterTabs << "\n";
            stream << "\n";
            stream << "\n";
            stream << "Average Settings:" << "\n";
            stream << "Pre stim: " << iPreStimSamples << "\n";
            stream << "Post stim: " << iPostStimSamples << "\n";
            stream << "Baseline from: " << iBaselineFromSamples << "\n";
            stream << "Baseline to: " << iBaselineToSamples << "\n";
            stream << "Number Averages: " << iNumAverages << "\n";
            stream << "Stim Channels: " << sStimChan << "\n";
            stream << "\n";
            stream << "\n";
            stream << "Source Space:" << "\n";
            stream << "Number Spaces: " << pFwdSolution->src.size() << "\n";
            stream << "Number Sources: " << pFwdSolution->nsource << "\n";
            stream << "Number Sources Clustered: " << pFwdSolution->nsource << "\n";
            stream << "\n";
            stream << "\n";
            stream << "HPI:" << "\n";
            stream << "Number coils: " << vecFreqs.size() << "\n";
        }
    }

    //=============================================================================================================
    // actual pipeline

    QElapsedTimer timer;
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

            // update head position if good fit
            if(dMeanErrorDist < dErrorMax) {
                HPIFit::storeHeadPosition(first/pFiffInfo->sfreq, fitResult.devHeadTrans.trans, matPosition, fitResult.GoF, fitResult.errorDistances);

                // update 3D View (not working yet)
                update3DView(fitResult);

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
            }

            // update Forward solution
            if(bDoHeadPosUpdate) {
                if(fitResult.bIsLargeHeadMovement) {
                    timer.start();
                    transMegHeadOld = fitResult.devHeadTrans.toOld();
                    pComputeFwd->updateHeadPos(&transMegHeadOld);           // update
                    pFwdSolution->sol = pComputeFwd->sol;                   // store results
                    pFwdSolution->sol_grad = pComputeFwd->sol_grad;
                    // cluster
                    pClusteredFwd = MNEForwardSolution::SPtr(new MNEForwardSolution(pFwdSolution->cluster_forward_solution(*pAnnotationSet.data(), 200, defaultD, noiseCov/*fiffComputedCov*/)));
                    forwardMeg = pClusteredFwd->pick_types(true, false);            // only take meg solution
                    vecTimeUpdate(i) = timer.elapsed();
//                    if(!fiffComputedCov.isEmpty()) {
//                    invOp = MNEInverseOperator(*pFiffInfoCompute.data(),                   // create new inverse operator
//                                               forwardMeg,
//                                               fiffComputedCov,
//                                               0.2f,
//                                               0.8f);
//                    }

                    invOp = MNEInverseOperator(*pFiffInfoCompute.data(),                   // create new inverse operator
                                                                   forwardMeg,
                                                                   noiseCov,
                                                                   0.2f,
                                                                   0.8f);
                    m_pFiffInfoForward = FIFFLIB::FiffInfoBase::SPtr(new FIFFLIB::FiffInfoBase(forwardMeg.info));       // update forward fiff info
    //                pRtInvOp->setFwdSolution(pFwdSolution);
    //                pRtInvOp->append(fiffComputedCov);
                }
            }
        }
        // Filtering
        matData = matSparseProjMult * matData;
        QList<FilterKernel> list;
        list << filterKernel;
        matData = pRtFilter->filterData(matData,
                                        list,
                                        lFilterChannelList);

//        // Covariance
//        fiffCov = rtCov.estimateCovariance(matData, iEstimationSamples);        // updates once iEstimationSamples is exceeded
//        if(!fiffCov.names.isEmpty()) {
//            fiffComputedCov = fiffCov;
////            pRtInvOp->append(fiffComputedCov);
//            m_qListCovChNames = fiffComputedCov.names;
//            invOp = MNEInverseOperator(*pFiffInfoCompute.data(),                // create new inverse operator
//                                       forwardMeg,
//                                       fiffComputedCov,
//                                       0.2f,
//                                       0.8f);
//            qDebug() << "Covariance updated";
//        }

        // averaging
        // Init the stim channels
        for(qint32 i = 0; i < pFiffInfo->chs.size(); ++i) {
            if(pFiffInfo->chs[i].kind == FIFFV_STIM_CH) {
                mapStimChsIndexNames.insert(pFiffInfo->chs[i].ch_name,i);
            }
        }
        pRtAve->append(matData);
        QObject::connect(pRtAve.data(), &RtAve::evokedStim, onNewEvokedSet);
        evokedSet = m_evokedSet;
        lResponsibleTriggerTypes = m_lResponsibleTriggerTypes;

        // source estimate
        if(evokedSet.evoked.size()>0 && !noiseCov.names.isEmpty()) {
            // only if evoked and noise cov
            currentEvoked = evokedSet.evoked.at(0);
            FiffEvoked megEvoked = currentEvoked.pick_channels(pickedChannels);
            m_pFiffInfoInput = QSharedPointer<FiffInfo>(new FiffInfo(megEvoked.info));

            calcFiffInfo();     // prepare fiff info from different objects (ToDo: only do once)

            // IOUtils::write_eigen_matrix(megEvoked.data, QCoreApplication::applicationDirPath() + "/MNE-sample-data/result.txt");

            MinimumNorm minimumNorm(invOp, lambda2, sMethod);
            MNESourceEstimate sourceEstimate = minimumNorm.calculateInverse(currentEvoked);
            if(!sourceEstimate.isEmpty()) {
//                std::cout << "\nsourceEstimate:\n" << sourceEstimate.data.block(0,0,10,10) << std::endl;
//                qDebug() << sourceEstimate.data.rows() << "x" << sourceEstimate.data.cols();
                if(bDoLogging) {
                    QFile fileSTC(sCurrentDir + "/" + QString::number(i)  + "_" + sID + "-vol.stc");
                    sourceEstimate.write(fileSTC);
                    QFile fileInvOp(sCurrentDir + "/" + QString::number(i)  + "_" + sID + "-inv.fif");
                    invOp.write(fileInvOp);
                    IOUtils::write_eigen_matrix(currentEvoked.data, sCurrentDir + "/" + QString::number(i)  + "_" + sID + "evoked.txt");
                }
            }
        }
        if(bWriteFilteredData) {
            // Write Data
            if(first_buffer) {
                if(first > 0) {
                    outfid->write_int(FIFF_FIRST_SAMPLE, &first);
                }
                first_buffer = false;
            }
            outfid->write_raw_buffer(matData,cals);
        }
    }

    // ready
    outfid->finish_writing_raw();

    if(MneDataTreeItem* pRTDataItem = m_p3DDataModel->addSourceData("Subject",
                                                                   "left auditory",
                                                                   sourceEstimate,
                                                                   forwardMeg,
                                                                   *pSurfaceSet.data(),
                                                                   *pAnnotationSet.data())) {
        pRTDataItem->setLoopState(true);
        pRTDataItem->setTimeInterval(17);
        pRTDataItem->setNumberAverages(1);
        pRTDataItem->setStreamingState(true);
        pRTDataItem->setThresholds(QVector3D(0.0,0.5,10.0));
        pRTDataItem->setVisualizationType("Interpolation based");
        pRTDataItem->setColormapType("Hot");
        pRTDataItem->setAlpha(0.75f);
        pRTDataItem->setTransform(mriHeadTrans);
        pRTDataItem->applyTransform(fitResult.devHeadTrans);
    }

    p3DAbstractView->show();

    // write hpi results
    if(bDoLogging) {
        QString sHPIFile = sID + "_hpi.txt";
        QString sTimeFile = sID + "_updateTime.txt";
        IOUtils::write_eigen_matrix(matPosition, sCurrentDir + "/" + sHPIFile);
        IOUtils::write_eigen_matrix(vecTimeUpdate, sCurrentDir + "/" + sTimeFile);
    }

    qDebug() << "Done";
    return a.exec();;
}
