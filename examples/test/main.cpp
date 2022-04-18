//=============================================================================================================
/**
 * @file     main.cpp
 * @author   1 <2>
 * @since    0.1.0
 * @date     3, 4
 *
 * @section  LICENSE
 *
 * Copyright (C) 4, 1. All rights reserved.
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
 * @brief     5.
 *
 */

//=============================================================================================================
// INCLUDES
//=============================================================================================================

#include <utils/generics/applicationlogger.h>
#include <utils/ioutils.h>
#include <disp/plots/plot.h>

//=============================================================================================================
// QT INCLUDES
//=============================================================================================================

#include <QApplication>
#include <QCommandLineParser>
#include <iostream>
#include <iostream>

//=============================================================================================================
// Eigen
//=============================================================================================================
#include <qmath.h>

//=============================================================================================================
// USED NAMESPACES
//=============================================================================================================
using namespace Eigen;
using namespace UTILSLIB;
using namespace DISPLIB;

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

int m_iFittingWindowSize = 150;

void runBuffer(MatrixXd matData) {

    int iDataLength = matData.cols();
    int iDataChannels = matData.rows();
    int iNTests = 5;
    int iDataIndexCounter = 0;
    int iStoreCounter = 0;
    int iFittingWindowSize = 0;

    MatrixXd matDataMerged(iDataChannels, iFittingWindowSize);

    for(int i = 0; i< iNTests; i ++) {
        // if fitting window size changed, reset counter and resize matDataMerged
        if(iFittingWindowSize != m_iFittingWindowSize) {
            iFittingWindowSize = m_iFittingWindowSize;
            qDebug() << "Fitting window size: " << iFittingWindowSize << "\n";
            matDataMerged.resize(iDataChannels, iFittingWindowSize);
            iDataIndexCounter = 0;
        }
        // count until we have enough samples and fill up matDataMerged
        if(iDataIndexCounter + matData.cols() < matDataMerged.cols()) {
            matDataMerged.block(0, iDataIndexCounter, matData.rows(), matData.cols()) = matData;
            iDataIndexCounter += matData.cols();
            qDebug() << "Data in counter:" << iDataIndexCounter;
        } else {
            qDebug() << "Writing data.";
            matDataMerged.block(0, iDataIndexCounter, matData.rows(), matDataMerged.cols()-iDataIndexCounter) =
                matData.block(0, 0, matData.rows(), matDataMerged.cols()-iDataIndexCounter);
            iStoreCounter++;

            // write to file
            std::string sFilePath = std::string("C:/Git/mne-cpp/bin/MNE-sample-data/sine") + std::to_string(iFittingWindowSize) + std::to_string(iDataLength) + std::to_string(iStoreCounter) + std::string(".txt");
            IOUtils::write_eigen_matrix(matDataMerged,sFilePath);
            iDataIndexCounter = 0;
        }
    }
}

int main(int argc, char *argv[])
{
    qInstallMessageHandler(UTILSLIB::ApplicationLogger::customLogWriter);
    QApplication a(argc, argv);

    // Command Line Parser
    QCommandLineParser parser;
    parser.setApplicationDescription("Example name");
    parser.addHelpOption();

    QCommandLineOption parameterOption("parameter", "The first parameter description.");

    parser.addOption(parameterOption);

    parser.process(a);

    // Add exampel code here

    int iSines = 1;
    int iDataSize = 100;
    double dSineFreq = 1.0;

    MatrixXd matData(iSines,iDataSize);
    const VectorXd vecTime = VectorXd::LinSpaced(iDataSize, 0, iDataSize-1) *1.0/iDataSize; // 1s
    for(int i = 0; i < iSines; i++) {
        matData.row(i) = sin(2*M_PI*dSineFreq*vecTime.array());
    }

    m_iFittingWindowSize = 150;
    runBuffer(matData);
    m_iFittingWindowSize = 200;
    runBuffer(matData);
    m_iFittingWindowSize = 300;
    runBuffer(matData);

    return a.exec();
}
