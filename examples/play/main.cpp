//=============================================================================================================
/**
 * @file     main.cpp
 * @author   a <b>
 * @since    0.1.0
 * @date     c, d
 *
 * @section  LICENSE
 *
 * Copyright (C) d, a. All rights reserved.
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
 * @brief     e.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <iostream>
#include <utils/generics/applicationlogger.h>
#include <fiff/fiff_coord_trans.h>
//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QApplication>
#include <QCommandLineParser>

//=============================================================================================================
// Eigen
//=============================================================================================================

#include <Eigen/Dense>

#define M_PI       3.14159265358979323846

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================
using namespace Eigen;
using namespace FIFFLIB;
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


double objFun2(const Eigen::MatrixXd &matTrans,
              const Eigen::MatrixXd &matCoilDev,
              const Eigen::MatrixXd &matCoilHead)
{
    /*
     *      Follows implementation from mne-python
     *      https://github.com/mne-tools/mne-python/blob/59c3b921cf7609d0f2ffc19c1a7dc276110100a3/mne/chpi.py#L500
     *
     */

    double dDenom = 0.0;
    double dGof = 0.0;
    double dSquaredSum = 0.0;
    MatrixXd matRot = matTrans.block(0,0,3,3);
    VectorXd vecTrans = matTrans.block<3,1>(0,3);

    MatrixXd matX = matCoilDev*matRot; // apply rotation
    matX = matX.rowwise() + vecTrans.transpose(); // apply translation
    matX -= matCoilHead; // substract target
    matX = matX.cwiseProduct(matX); // square values
    dSquaredSum = matX.sum(); // sum up

    VectorXd vecCenter = matCoilHead.colwise().mean(); // center of gravity
    dDenom = (matCoilHead.rowwise() - vecCenter.transpose()).norm(); // substract center of gravity -> centralize and take norm
    dDenom *= dDenom; // square

    dGof = 1.0 - dSquaredSum/dDenom;

    return dGof;
}

Eigen::Matrix4d computeTransformation(Eigen::MatrixXd matNH, MatrixXd matBT)
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

    // Add exampel code here

    MatrixXd matCoilDev(4,3);
    matCoilDev << 1.0,0.0,0.0,
                  1.0,1.0,0.0,
                  1.0,1.0,1.0,
                  1.0,0.0,1.0;

    MatrixXd matCoilHead = matCoilDev;
    matCoilHead.row(2) = matCoilDev.row(1);
    matCoilHead.row(1) = matCoilDev.row(2);

    std::cout << "Head" << std::endl;
    std::cout << matCoilHead << std::endl;
    std::cout << "Dev" << std::endl;
    std::cout << matCoilDev << std::endl;


    // get started

    int iCoils = matCoilDev.rows();
    std::vector<int> vecOrder(iCoils);  // use std container because we use std::permutation later on
    std::vector<int> vecBestOrder(iCoils);
    for(int i = 0; i < iCoils; i++) {
        vecOrder[i] = i;
    }

    MatrixXd matTempHead = matCoilHead;
    MatrixXd matTrans = MatrixXd::Identity(4,4);
    MatrixXd matBestTrans = MatrixXd::Identity(4,4);
    double dBestG = -999.0; // can be negative quite often

    FiffCoordTrans transRef;
    float fAngle = 0.0;

    // start permutation
    do {
        // reorder
        for(int i = 0; i < iCoils; i++) {
            matTempHead.row(i) = matCoilHead.row(vecOrder[i]);
        }
        std::cout << "Order " << vecOrder[0] << vecOrder[1]<< vecOrder[2]<< vecOrder[3] << std::endl;
        std::cout << "Temp" << std::endl;
        std::cout << matTempHead << std::endl;

        matTrans = computeTransformation(matTempHead,matCoilDev);
        double dGof = objFun2(matTrans,matTempHead,matCoilDev);

        // Penelaize by heavy rotation
        fAngle = transRef.angleTo(matTrans.cast<float>());
        dGof = std::pow(dGof* std::max(1.0-fAngle/M_PI,0.0) ,0.25);

        if(dGof > dBestG) {
            dBestG = dGof;
            vecBestOrder = vecOrder;
            matBestTrans = matTrans;
        }

    } while (std::next_permutation(vecOrder.begin(), vecOrder.end()));

    std::cout << vecBestOrder[0]
              << vecBestOrder[1]
              << vecBestOrder[2]
              << vecBestOrder[3]
              << std::endl;

    return a.exec();
}
