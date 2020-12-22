//=============================================================================================================
/**
 * @file     hpifit.cpp
 * @author   Lorenz Esch <lesch@mgh.harvard.edu>;
 *           Ruben Dörfel <ruben.doerfel@tu-ilmenau.de>;
 *           Matti Hamalainen <msh@nmr.mgh.harvard.edu>
 * @since    0.1.0
 * @date     March, 2017
 *
 * @section  LICENSE
 *
 * Copyright (C) 2017, Lorenz Esch, Matti Hamalainen. All rights reserved.
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
 * @brief    HPIFit class defintion.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include "hpifit.h"
#include "hpifitdata.h"

#include <utils/ioutils.h>
#include <utils/mnemath.h>

#include <iostream>
#include <fiff/fiff_cov.h>
#include <fiff/fiff_dig_point_set.h>
#include <fstream>

#include <fwd/fwd_coil_set.h>

//=============================================================================================================
// EIGEN INCLUDES
//=============================================================================================================

#include <Eigen/Dense>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QFuture>
#include <QtConcurrent/QtConcurrent>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================

using namespace Eigen;
using namespace INVERSELIB;
using namespace FIFFLIB;
using namespace FWDLIB;

//=============================================================================================================
// DEFINE GLOBAL METHODS
//=============================================================================================================

//=============================================================================================================
// DEFINE MEMBER METHODS
//=============================================================================================================

HPIFit::HPIFit(FiffInfo::SPtr pFiffInfo,
               bool bDoFastFit)
    : m_bDoFastFit(bDoFastFit)
{
    // init member variables
    m_lChannels = QList<FIFFLIB::FiffChInfo>();
    m_vecInnerind = QVector<int>();
    m_sensors = SensorSet();
    m_lBads = pFiffInfo->bads;
    m_matModel = MatrixXd(0,0);
    m_vecFreqs = QVector<int>();

    // init coils
    m_coilTemplate = NULL;
    m_coilMeg = NULL;

    updateChannels(pFiffInfo);
    updateSensor();
}

//=============================================================================================================

void HPIFit::fitHPI(const MatrixXd& t_mat,
                    const MatrixXd& t_matProjectors,
                    FiffCoordTrans& transDevHead,
                    const QVector<int>& vFreqs,
                    QVector<double>& vGof,
                    VectorXd& vecGoF,
                    FiffDigPointSet& fittedPointSet,
                    FiffInfo::SPtr pFiffInfo,
                    bool bDoDebug,
                    const QString& sHPIResourceDir,
                    int iMaxIterations,
                    float fAbortError)
{
    //Check if data was passed
    if(t_mat.rows() == 0 || t_mat.cols() == 0 ) {
        std::cout<<std::endl<< "HPIFit::fitHPI - No data passed. Returning.";
    }

    //Check if projector was passed
    if(t_matProjectors.rows() == 0 || t_matProjectors.cols() == 0 ) {
        std::cout<<std::endl<< "HPIFit::fitHPI - No projector passed. Returning.";
    }

    vGof.clear();

    struct SensorInfo sensors;
    struct CoilParam coil;
    int numCh = pFiffInfo->nchan;
    int samF = pFiffInfo->sfreq;
    int samLoc = t_mat.cols(); // minimum samples required to localize numLoc times in a second

    //Get HPI coils from digitizers and set number of coils
    int numCoils = 0;
    QList<FiffDigPoint> lHPIPoints;

    for(int i = 0; i < pFiffInfo->dig.size(); ++i) {
        if(pFiffInfo->dig[i].kind == FIFFV_POINT_HPI) {
            numCoils++;
            lHPIPoints.append(pFiffInfo->dig[i]);
        }
    }

    //Set coil frequencies
    Eigen::VectorXd coilfreq(numCoils);

    if(vFreqs.size() >= numCoils) {
        for(int i = 0; i < numCoils; ++i) {
            coilfreq[i] = vFreqs.at(i);
            //std::cout<<std::endl << coilfreq[i] << "Hz";
        }
    } else {
        std::cout<<std::endl<< "HPIFit::fitHPI - Not enough coil frequencies specified. Returning.";
        return;
    }

    // Initialize HPI coils location and moment
    coil.pos = Eigen::MatrixXd::Zero(numCoils,3);
    coil.mom = Eigen::MatrixXd::Zero(numCoils,3);
    coil.dpfiterror = Eigen::VectorXd::Zero(numCoils);
    coil.dpfitnumitr = Eigen::VectorXd::Zero(numCoils);

    // Generate simulated data
    Eigen::MatrixXd simsig(samLoc,numCoils*2);
    Eigen::VectorXd time(samLoc);

    for (int i = 0; i < samLoc; ++i) {
        time[i] = i*1.0/samF;
    }

    for(int i = 0; i < numCoils; ++i) {
        for(int j = 0; j < samLoc; ++j) {
            simsig(j,i) = sin(2*M_PI*coilfreq[i]*time[j]);
            simsig(j,i+numCoils) = cos(2*M_PI*coilfreq[i]*time[j]);
        }
    }

    // Create digitized HPI coil position matrix
    Eigen::MatrixXd headHPI(numCoils,3);

    // check the pFiffInfo->dig information. If dig is empty, set the headHPI is 0;
    if (lHPIPoints.size() > 0) {
        for (int i = 0; i < lHPIPoints.size(); ++i) {
            headHPI(i,0) = lHPIPoints.at(i).r[0];
            headHPI(i,1) = lHPIPoints.at(i).r[1];
            headHPI(i,2) = lHPIPoints.at(i).r[2];
        }
    } else {
        for (int i = 0; i < numCoils; ++i) {
            headHPI(i,0) = 0;
            headHPI(i,1) = 0;
            headHPI(i,2) = 0;
        }
    }

    // Get the indices of inner layer channels and exclude bad channels.
    //TODO: Only supports babymeg and vectorview gradiometeres for hpi fitting.
    QVector<int> innerind(0);

    for (int i = 0; i < numCh; ++i) {
        if(pFiffInfo->chs[i].chpos.coil_type == FIFFV_COIL_BABY_MAG ||
            pFiffInfo->chs[i].chpos.coil_type == FIFFV_COIL_VV_PLANAR_T1 ||
            pFiffInfo->chs[i].chpos.coil_type == FIFFV_COIL_VV_PLANAR_T2 ||
            pFiffInfo->chs[i].chpos.coil_type == FIFFV_COIL_VV_PLANAR_T3) {
            // Check if the sensor is bad, if not append to innerind
            if(!(pFiffInfo->bads.contains(pFiffInfo->ch_names.at(i)))) {
                innerind.append(i);
            }
        }
    }

    //Create new projector based on the excluded channels, first exclude the rows then the columns
    MatrixXd matProjectorsRows(innerind.size(),t_matProjectors.cols());
    MatrixXd matProjectorsInnerind(innerind.size(),innerind.size());

    for (int i = 0; i < matProjectorsRows.rows(); ++i) {
        matProjectorsRows.row(i) = t_matProjectors.row(innerind.at(i));
    }

    for (int i = 0; i < matProjectorsInnerind.cols(); ++i) {
        matProjectorsInnerind.col(i) = matProjectorsRows.col(innerind.at(i));
    }

    //UTILSLIB::IOUtils::write_eigen_matrix(matProjectorsInnerind, "matProjectorsInnerind.txt");
    //UTILSLIB::IOUtils::write_eigen_matrix(t_matProjectors, "t_matProjectors.txt");

    // Initialize inner layer sensors
    sensors.coilpos = Eigen::MatrixXd::Zero(innerind.size(),3);
    sensors.coilori = Eigen::MatrixXd::Zero(innerind.size(),3);
    sensors.tra = Eigen::MatrixXd::Identity(innerind.size(),innerind.size());

    for(int i = 0; i < innerind.size(); i++) {
        sensors.coilpos(i,0) = pFiffInfo->chs[innerind.at(i)].chpos.r0[0];
        sensors.coilpos(i,1) = pFiffInfo->chs[innerind.at(i)].chpos.r0[1];
        sensors.coilpos(i,2) = pFiffInfo->chs[innerind.at(i)].chpos.r0[2];
        sensors.coilori(i,0) = pFiffInfo->chs[innerind.at(i)].chpos.ez[0];
        sensors.coilori(i,1) = pFiffInfo->chs[innerind.at(i)].chpos.ez[1];
        sensors.coilori(i,2) = pFiffInfo->chs[innerind.at(i)].chpos.ez[2];
    }

    Eigen::MatrixXd topo(innerind.size(), numCoils*2);
    Eigen::MatrixXd amp(innerind.size(), numCoils);
    Eigen::MatrixXd ampC(innerind.size(), numCoils);

    // Get the data from inner layer channels
    Eigen::MatrixXd innerdata(innerind.size(), t_mat.cols());

    for(int j = 0; j < innerind.size(); ++j) {
        innerdata.row(j) << t_mat.row(innerind[j]);
    }

    // Calculate topo
    topo = innerdata * UTILSLIB::MNEMath::pinv(simsig).transpose(); // topo: # of good inner channel x 8

    // Select sine or cosine component depending on the relative size
    amp  = topo.leftCols(numCoils); // amp: # of good inner channel x 4
    ampC = topo.rightCols(numCoils);

    for(int j = 0; j < numCoils; ++j) {
        float nS = 0.0;
        float nC = 0.0;
        for(int i = 0; i < innerind.size(); ++i) {
            nS += amp(i,j)*amp(i,j);
            nC += ampC(i,j)*ampC(i,j);
        }

        if(nC > nS) {
            for(int i = 0; i < innerind.size(); ++i) {
                amp(i,j) = ampC(i,j);
            }
        }
    }

    //Find good seed point/starting point for the coil position in 3D space
    //Find biggest amplitude per pickup coil (sensor) and store corresponding sensor channel index
    VectorXi chIdcs(numCoils);

    for (int j = 0; j < numCoils; j++) {
        double maxVal = 0;
        int chIdx = 0;

        for (int i = 0; i < amp.rows(); ++i) {
            if(std::fabs(amp(i,j)) > maxVal) {
                maxVal = std::fabs(amp(i,j));

                if(chIdx < innerind.size()) {
                    chIdx = innerind.at(i);
                }
            }
        }

        chIdcs(j) = chIdx;
    }

    //Generate seed point by projection the found channel position 3cm inwards
    Eigen::MatrixXd coilPos = Eigen::MatrixXd::Zero(numCoils,3);

    for (int j = 0; j < chIdcs.rows(); ++j) {
        int chIdx = chIdcs(j);

        if(chIdx < pFiffInfo->chs.size()) {
            double x = pFiffInfo->chs.at(chIdcs(j)).chpos.r0[0];
            double y = pFiffInfo->chs.at(chIdcs(j)).chpos.r0[1];
            double z = pFiffInfo->chs.at(chIdcs(j)).chpos.r0[2];

            coilPos(j,0) = -1 * pFiffInfo->chs.at(chIdcs(j)).chpos.ez[0] * 0.03 + x;
            coilPos(j,1) = -1 * pFiffInfo->chs.at(chIdcs(j)).chpos.ez[1] * 0.03 + y;
            coilPos(j,2) = -1 * pFiffInfo->chs.at(chIdcs(j)).chpos.ez[2] * 0.03 + z;
        }

        //std::cout << "HPIFit::fitHPI - Coil " << j << " max value index " << chIdx << std::endl;
    }

    coil.pos = coilPos;

    coil = dipfit(coil, sensors, amp, numCoils, matProjectorsInnerind);

    Eigen::Matrix4d trans = computeTransformation(headHPI, coil.pos);
    //Eigen::Matrix4d trans = computeTransformation(coil.pos, headHPI);

    // Store the final result to fiff info
    // Set final device/head matrix and its inverse to the fiff info
    transDevHead.from = 1;
    transDevHead.to = 4;

    for(int r = 0; r < 4; ++r) {
        for(int c = 0; c < 4 ; ++c) {
            transDevHead.trans(r,c) = trans(r,c);
        }
    }

    // Also store the inverse
    transDevHead.invtrans = transDevHead.trans.inverse();

    //Calculate GOF
    MatrixXd temp = coil.pos;
    temp.conservativeResize(coil.pos.rows(),coil.pos.cols()+1);

    temp.block(0,3,numCoils,1).setOnes();
    temp.transposeInPlace();

    MatrixXd testPos = trans * temp;
    MatrixXd diffPos = testPos.block(0,0,3,numCoils) - headHPI.transpose();

    for(int i = 0; i < diffPos.cols(); ++i) {
        vGof.append(diffPos.col(i).norm());
    }

    //Generate final fitted points and store in digitizer set
    for(int i = 0; i < coil.pos.rows(); ++i) {
        FiffDigPoint digPoint;
        digPoint.kind = FIFFV_POINT_EEG;
        digPoint.ident = i;
        digPoint.r[0] = coil.pos(i,0);
        digPoint.r[1] = coil.pos(i,1);
        digPoint.r[2] = coil.pos(i,2);

        fittedPointSet << digPoint;
    }

    if(bDoDebug) {
        // DEBUG HPI fitting and write debug results
        QString sTimeStamp = QDateTime::currentDateTime().toString("yyMMdd_hhmmss");

        if(!QDir(sHPIResourceDir).exists()) {
            QDir().mkdir(sHPIResourceDir);
        }

        UTILSLIB::IOUtils::write_eigen_matrix(sensors.coilpos, QString("%1/%2_coilPosOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));

        UTILSLIB::IOUtils::write_eigen_matrix(coilPos, QString("%1/%2_coilPosSeedOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));

        UTILSLIB::IOUtils::write_eigen_matrix(coil.pos, QString("%1/%2_hpiPosOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));

        UTILSLIB::IOUtils::write_eigen_matrix(headHPI, QString("%1/%2_headHPIOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));

        MatrixXd testPosCut = testPos.transpose();//block(0,0,3,4);
        UTILSLIB::IOUtils::write_eigen_matrix(testPosCut, QString("%1/%2_testPosOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));

        MatrixXi idx_mat(chIdcs.rows(),1);
        std::cout << chIdcs << std::endl;
        idx_mat.col(0) = chIdcs;
        UTILSLIB::IOUtils::write_eigen_matrix(idx_mat, QString("%1/%2_idxOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));

        MatrixXd coilFreq_mat(coilfreq.rows(),1);
        coilFreq_mat.col(0) = coilfreq;
        UTILSLIB::IOUtils::write_eigen_matrix(coilFreq_mat, QString("%1/%2_coilFreqOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));

        UTILSLIB::IOUtils::write_eigen_matrix(diffPos, QString("%1/%2_diffPosOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));

        UTILSLIB::IOUtils::write_eigen_matrix(amp, QString("%1/%2_ampOld_mat").arg(sHPIResourceDir).arg(sTimeStamp));
    }
}

//=============================================================================================================

void HPIFit::findOrder(const MatrixXd& t_mat,
                       const MatrixXd& t_matProjectors,
                       FiffCoordTrans& transDevHead,
                       QVector<int>& vecFreqs,
                       QVector<double>& vecError,
                       VectorXd& vecGoF,
                       FiffDigPointSet& fittedPointSet,
                       FiffInfo::SPtr pFiffInfo)
{
    // create temporary copies that are necessary to reset values that are passed to fitHpi()
    fittedPointSet.clear();
    transDevHead.clear();
    vecError.fill(0);

    FiffDigPointSet fittedPointSetTemp = fittedPointSet;
    FiffCoordTrans transDevHeadTemp = transDevHead;
    FiffInfo::SPtr pFiffInfoTemp = pFiffInfo;
    QVector<int> vecToOrder = vecFreqs;
    QVector<int> vecFreqTemp(vecFreqs.size());
    QVector<double> vecErrorTemp = vecError;
    VectorXd vecGoFTemp = vecGoF;
    bool bIdentity = false;

    MatrixXf matTrans = transDevHead.trans;
    if(transDevHead.trans == MatrixXf::Identity(4,4).cast<float>()) {
        // avoid identity since this leads to problems with this method in fitHpi.
        // the hpi fit is robust enough to handle bad starting points
        transDevHeadTemp.trans(3,0) = 0.000001;
        bIdentity = true;
    }

    // perform vecFreqs.size() hpi fits with same frequencies in each iteration
    for(int i = 0; i < vecFreqs.size(); i++){
        vecFreqTemp.fill(vecFreqs[i]);

        // hpi Fit
        fitHPI(t_mat, t_matProjectors, transDevHeadTemp, vecFreqTemp, vecErrorTemp, vecGoFTemp, fittedPointSetTemp, pFiffInfoTemp);

        // get location of maximum GoF -> correct assignment of coil - frequency
        VectorXd::Index indMax;
        vecGoFTemp.maxCoeff(&indMax);
        vecToOrder[indMax] = vecFreqs[i];

        // std::cout << vecGoFTemp[0] << " " << vecGoFTemp[1] << " " << vecGoFTemp[2] << " " << vecGoFTemp[3] << " " << std::endl;

        // reset values that are edidet by fitHpi()
        fittedPointSetTemp = fittedPointSet;
        pFiffInfoTemp = pFiffInfo;
        transDevHeadTemp = transDevHead;

        if(bIdentity) {
            transDevHeadTemp.trans(3,0) = 0.000001;
        }
        vecErrorTemp = vecError;
        vecGoFTemp = vecGoF;
    }
    // check if still all frequencies are represented and update model
    if(std::accumulate(vecFreqs.begin(), vecFreqs.end(), .0) ==  std::accumulate(vecToOrder.begin(), vecToOrder.end(), .0)) {
        vecFreqs = vecToOrder;
    } else {
        qWarning() << "HPIFit::findOrder: frequencie ordering went wrong";
    }
    qInfo() << "HPIFit::findOrder: vecFreqs = " << vecFreqs;
}

//=============================================================================================================

CoilParam HPIFit::dipfit(struct CoilParam coil, struct SensorInfo sensors, const Eigen::MatrixXd& data, int numCoils, const Eigen::MatrixXd& t_matProjectors)
{
    //Do this in conncurrent mode
    //Generate QList structure which can be handled by the QConcurrent framework
    QList<HPIFitData> lCoilData;

    for(qint32 i = 0; i < numCoils; ++i) {
        HPIFitData coilData;
        coilData.coilPos = coil.pos.row(i);
        coilData.sensorData = data.col(i);
        coilData.sensorPos = sensors;
        coilData.matProjector = t_matProjectors;

        lCoilData.append(coilData);
    }
    //Do the concurrent filtering
    if(!lCoilData.isEmpty()) {
        //        //Do sequential
        //        for(int l = 0; l < lCoilData.size(); ++l) {
        //            doDipfitConcurrent(lCoilData[l]);
        //        }

        //Do concurrent
        QFuture<void> future = QtConcurrent::map(lCoilData,
                                                 &HPIFitData::doDipfitConcurrent);
        future.waitForFinished();

        //Transform results to final coil information
        for(qint32 i = 0; i < lCoilData.size(); ++i) {
            coil.pos.row(i) = lCoilData.at(i).coilPos;
            coil.mom = lCoilData.at(i).errorInfo.moment.transpose();
            coil.dpfiterror(i) = lCoilData.at(i).errorInfo.error;
            coil.dpfitnumitr(i) = lCoilData.at(i).errorInfo.numIterations;

            //std::cout<<std::endl<< "HPIFit::dipfit - Itr steps for coil " << i << " =" <<coil.dpfitnumitr(i);
        }
    }

    return coil;
}

//=============================================================================================================

Eigen::Matrix4d HPIFit::computeTransformation(Eigen::MatrixXd matNH, MatrixXd matBT)
{
    MatrixXd matXdiff, matYdiff, matZdiff, matC, matQ;
    Matrix4d matTransFinal = Matrix4d::Identity(4,4);
    Matrix4d matRot = Matrix4d::Zero(4,4);
    Matrix4d matTrans = Matrix4d::Identity(4,4);
    double dMeanX,dMeanY,dMeanZ,dNormf;

    for(int i = 0; i < 15; ++i) {
        // Calculate mean translation for all points -> centroid of both data sets
        matXdiff = matNH.col(0) - matBT.col(0);
        matYdiff = matNH.col(1) - matBT.col(1);
        matZdiff = matNH.col(2) - matBT.col(2);

        dMeanX = matXdiff.mean();
        dMeanY = matYdiff.mean();
        dMeanZ = matZdiff.mean();

        // Apply translation -> bring both data sets to the same center location
        for (int j = 0; j < matBT.rows(); ++j) {
            matBT(j,0) = matBT(j,0) + dMeanX;
            matBT(j,1) = matBT(j,1) + dMeanY;
            matBT(j,2) = matBT(j,2) + dMeanZ;
        }

        // Estimate rotation component
        matC = matBT.transpose() * matNH;

        JacobiSVD< MatrixXd > svd(matC ,Eigen::ComputeThinU | ComputeThinV);

        matQ = svd.matrixU() * svd.matrixV().transpose();

        //Handle special reflection case
        if(matQ.determinant() < 0) {
            matQ(0,2) = matQ(0,2) * -1;
            matQ(1,2) = matQ(1,2) * -1;
            matQ(2,2) = matQ(2,2) * -1;
        }

        // Apply rotation on translated points
        matBT = matBT * matQ;

        // Calculate GOF
        dNormf = (matNH.transpose()-matBT.transpose()).norm();

        // Store rotation part to transformation matrix
        matRot(3,3) = 1;
        for(int j = 0; j < 3; ++j) {
            for(int k = 0; k < 3; ++k) {
                matRot(j,k) = matQ(k,j);
            }
        }

        // Store translation part to transformation matrix
        matTrans(0,3) = dMeanX;
        matTrans(1,3) = dMeanY;
        matTrans(2,3) = dMeanZ;

        // Safe rotation and translation to final matrix for next iteration step
        // This step is safe to do since we change one of the input point sets (matBT)
        // ToDo: Replace this for loop with a least square solution process
        matTransFinal = matRot * matTrans * matTransFinal;
    }
    return matTransFinal;
}

//=============================================================================================================

void HPIFit::createSensorSet(SensorSet& sensors,
                             FwdCoilSet* coils)
{
    int iNchan = coils->ncoil;

    // init sensor struct
    int iNp = coils->coils[0]->np;
    sensors.w = RowVectorXd(iNchan*iNp);
    sensors.r0 = MatrixXd(iNchan,3);
    sensors.cosmag = MatrixXd(iNchan*iNp,3);
    sensors.rmag = MatrixXd(iNchan*iNp,3);
    sensors.ncoils = iNchan;
    sensors.tra = MatrixXd::Identity(iNchan,iNchan);
    sensors.np = iNp;

    for(int i = 0; i < iNchan; i++){
        FwdCoil* coil = (coils->coils[i]);
        MatrixXd matRmag = MatrixXd::Zero(iNp,3);
        MatrixXd matCosmag = MatrixXd::Zero(iNp,3);
        RowVectorXd vecW(iNp);

        sensors.r0(i,0) = coil->r0[0];
        sensors.r0(i,1) = coil->r0[1];
        sensors.r0(i,2) = coil->r0[2];

        for (int p = 0; p < iNp; p++){
            sensors.w(i*iNp+p) = coil->w[p];
            for (int c = 0; c < 3; c++) {
                matRmag(p,c)   = coil->rmag[p][c];
                matCosmag(p,c) = coil->cosmag[p][c];
            }
        }

        sensors.cosmag.block(i*iNp,0,iNp,3) = matCosmag;
        sensors.rmag.block(i*iNp,0,iNp,3) = matRmag;
    }
}

//=============================================================================================================

void HPIFit::storeHeadPosition(float fTime,
                               const Eigen::MatrixXf& transDevHead,
                               Eigen::MatrixXd& matPosition,
                               const Eigen::VectorXd& vecGoF,
                               const QVector<double>& vecError)

{
    // Write quaternions and vecTime in position matrix. Format is the same like MaxFilter's .pos files.
    Matrix3f matRot = transDevHead.block(0,0,3,3);

    double dError = std::accumulate(vecError.begin(), vecError.end(), .0) / vecError.size();     // HPI estimation Error
    Eigen::Quaternionf quatHPI(matRot);

//    qDebug() << "quatHPI.x() " << "quatHPI.y() " << "quatHPI.y() " << "trans x " << "trans y " << "trans z ";
//    qDebug() << quatHPI.x() << quatHPI.y() << quatHPI.z() << transDevHead(0,3) << transDevHead(1,3) << transDevHead(2,3);

    matPosition.conservativeResize(matPosition.rows()+1, 10);
    matPosition(matPosition.rows()-1,0) = fTime;
    matPosition(matPosition.rows()-1,1) = quatHPI.x();
    matPosition(matPosition.rows()-1,2) = quatHPI.y();
    matPosition(matPosition.rows()-1,3) = quatHPI.z();
    matPosition(matPosition.rows()-1,4) = transDevHead(0,3);
    matPosition(matPosition.rows()-1,5) = transDevHead(1,3);
    matPosition(matPosition.rows()-1,6) = transDevHead(2,3);
    matPosition(matPosition.rows()-1,7) = vecGoF.mean();
    matPosition(matPosition.rows()-1,8) = dError;
    matPosition(matPosition.rows()-1,9) = 0;
}

//=============================================================================================================

void HPIFit::updateSensor()
{
    // Create MEG-Coils and read data
    int iAcc = 0;
    int iNch = m_lChannels.size();

    if(iNch == 0) {
        return;
    }

    FiffCoordTransOld* t = NULL;

    if(!m_coilTemplate) {
        // read coil_def.dat
        QString qPath = QString(QCoreApplication::applicationDirPath() + "/resources/general/coilDefinitions/coil_def.dat");
        m_coilTemplate = FwdCoilSet::read_coil_defs(qPath);
    }

    // create sensor set
    m_coilMeg = m_coilTemplate->create_meg_coils(m_lChannels, iNch, iAcc, t);
    createSensorSet(m_sensors, m_coilMeg);
}

//=============================================================================================================

void HPIFit::updateChannels(QSharedPointer<FIFFLIB::FiffInfo> pFiffInfo)
{
    // Get the indices of inner layer channels and exclude bad channels and create channellist
    int iNumCh = pFiffInfo->nchan;
    m_vecInnerind.clear();
    m_lChannels.clear();
    for (int i = 0; i < iNumCh; ++i) {
        if(pFiffInfo->chs[i].chpos.coil_type == FIFFV_COIL_BABY_MAG ||
           pFiffInfo->chs[i].chpos.coil_type == FIFFV_COIL_VV_PLANAR_T1 ||
           pFiffInfo->chs[i].chpos.coil_type == FIFFV_COIL_VV_PLANAR_T2 ||
           pFiffInfo->chs[i].chpos.coil_type == FIFFV_COIL_VV_PLANAR_T3) {
            // Check if the sensor is bad, if not append to innerind
            if(!(pFiffInfo->bads.contains(pFiffInfo->ch_names.at(i)))) {
                m_vecInnerind.append(i);
                m_lChannels.append(pFiffInfo->chs[i]);
            }
        }
    }

    m_lBads = pFiffInfo->bads;
}

//=============================================================================================================

void HPIFit::updateModel(const int iSamF,
                         const int iSamLoc,
                         int iLineF,
                         const QVector<int>& vecFreqs)
{
    int iNumCoils = vecFreqs.size();
    MatrixXd matSimsig;
    VectorXd vecTime = VectorXd::LinSpaced(iSamLoc, 0, iSamLoc-1) *1.0/iSamF;

    if(m_bDoFastFit){
        // Generate simulated data Matrix
        matSimsig.conservativeResize(iSamLoc,iNumCoils*2);

        for(int i = 0; i < iNumCoils; ++i) {
            matSimsig.col(i) = sin(2*M_PI*vecFreqs[i]*vecTime.array());
            matSimsig.col(i+iNumCoils) = cos(2*M_PI*vecFreqs[i]*vecTime.array());
        }
        m_matModel = UTILSLIB::MNEMath::pinv(matSimsig);
        return;

    } else {
        // add linefreq + harmonics + DC part to model
        matSimsig.conservativeResize(iSamLoc,iNumCoils*4);
        for(int i = 0; i < iNumCoils; ++i) {
            matSimsig.col(i) = sin(2*M_PI*vecFreqs[i]*vecTime.array());
            matSimsig.col(i+iNumCoils) = cos(2*M_PI*vecFreqs[i]*vecTime.array());
            matSimsig.col(i+2*iNumCoils) = sin(2*M_PI*iLineF*i*vecTime.array());
            matSimsig.col(i+3*iNumCoils) = cos(2*M_PI*iLineF*i*vecTime.array());
        }
        matSimsig.col(14) = RowVectorXd::LinSpaced(iSamLoc, -0.5, 0.5);
        matSimsig.col(15).fill(1);
    }
    m_matModel = UTILSLIB::MNEMath::pinv(matSimsig);

    // reorder for faster computation
    MatrixXd matTemp = m_matModel;
    RowVectorXi vecIndex(2*iNumCoils);
    vecIndex << 0,4,1,5,2,6,3,7;
    for(int i = 0; i < vecIndex.size(); ++i) {
        matTemp.row(i) = m_matModel.row(vecIndex(i));
    }
    m_matModel = matTemp;
}
