/*
FV3DMT by Suzuki Atsushi is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/
*/
#pragma once
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <time.h>
#include "BiCGSafe.h"
#include "DivergenceCorrection.h"
#include <omp.h>
#include "Output.h"
using namespace std;

inline std::complex<double> BiCGSafe::BiCGSafe::dot(Eigen::VectorXcd& a, Eigen::VectorXcd& b) {
    int n = a.size();

    //double valmax = 0.0;
    //for (int i = 0; i < n; i++) {
    //    if (valmax * valmax < a.coeff(i).real() * a.coeff(i).real() + a.coeff(i).imag() * a.coeff(i).imag()) {
    //        valmax = std::sqrt(a.coeff(i).real() * a.coeff(i).real() + a.coeff(i).imag() * a.coeff(i).imag());
    //    }
    //    if (valmax *valmax  < b.coeff(i).real() * b.coeff(i).real() + b.coeff(i).imag() * b.coeff(i).imag()) {
    //        valmax = std::sqrt(b.coeff(i).real() * b.coeff(i).real() + b.coeff(i).imag() * b.coeff(i).imag());
    //    }
    //}
    //if (valmax == 0.0) {
    //    valmax = 1.0;
    //}
    int numThreads = omp_get_max_threads();
    int numCalcPerThread = n / numThreads;
    vector<int> istart(numThreads);
    vector<int> iend(numThreads);
    vector<complex<double>> sums(numThreads);
    for (int i = 0; i < numThreads - 1; i++) {
        istart[i] = numCalcPerThread * i;
        iend[i] = numCalcPerThread * (i+1);
        sums[i] = 0.0;
    }
    istart[numThreads -1] = numCalcPerThread * (numThreads -1);
    iend[numThreads - 1] = n;
    sums[numThreads - 1] = 0.0;

    int k;
//#pragma omp parallel for private(k)
    for (int j = 0; j < numThreads; j++) {
        for (k = istart[j]; k < iend[j]; k++) {
            sums[j] += std::conj(a.coeff(k)) * b.coeff(k);
        }
    }


    complex<double> res = 0.0;
    for (int j = 0; j < numThreads; j++) {
        res += sums[j];
    }
    return res;
}
bool BiCGSafe::BiCGSafe::solve(Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* mat_in, Eigen::VectorXcd* x, Eigen::VectorXcd* rhs_in, double tol, bool isDividedByDiagonal) {
    //https://www.jstage.jst.go.jp/article/jsces/2005/0/2005_0_20050028/_pdf
    //BiCGSafe method based on minimization of associate residual
    //ê¸å`ï˚íˆéÆÇÃîΩïúâñ@ ä€ëP
    Eigen::setNbThreads(1);

    int n = mat_in->cols();

    bool isConverged = false;

    Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> mat = *mat_in;
    Eigen::VectorXcd rhs = *rhs_in;
    
    //if (isDividedByDiagonal) {
    //    for (int i = 0; i < n; i++) {
    //        for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
    //            int col = it.col();
    //            int row = it.row();
    //            if (row != col) {
    //                mat.coeffRef(row, col) /= mat.coeff(row, row);
    //            }

    //        }
    //        rhs.coeffRef(i) /= mat.coeff(i, i);
    //        mat.coeffRef(i, i) /= mat.coeff(i, i);
    //    }
    //}
    Eigen::DiagonalPreconditioner<std::complex<double>> precond;
    //Eigen::IdentityPreconditioner precond;
    //precond.setFillfactor(0);
    precond.compute(mat);

    //ILU::ILU precond(n);
    //precond.compute(&mat);


    int maxIters = m_maxIteration;

    int minIters = 10;

    int m_l = 1;
    int L = m_l;

    int k = -L;

    int iter = 0;
    m_iters = iter;
    


    Eigen::VectorXcd r0star(n);
    r0star.setZero();
    Eigen::VectorXcd x0star(n);
    x0star.setZero();
    const int seed = 12345;
    std::srand(seed);
    double randMax = RAND_MAX;



    //for (int j = 0; j < n; j++) {
    //    double pi = 3.14159265359;
    //    double radius = std::rand() / randMax;
    //    double theta = std::rand() / randMax * 2.0*pi;
    //    /*r0star.coeffRef(j).real(radius*std::cos(theta));
    //    r0star.coeffRef(j).imag(radius*std::sin(theta));*/
    //    r0star.coeffRef(j).real(rand());
    //    r0star.coeffRef(j).imag(rand());
    //}
    Eigen::VectorXcd r = rhs - mat * (*x);

    //r0star = rhs - mat * x0star;
    r0star = r;
    //r0star = r/std::sqrt(r.squaredNorm());

    //r0star = r0star/std::sqrt(r0star.dot(r0star).real());

    double r0_sqnorm = r.squaredNorm();



    Eigen::VectorXcd x0 = *x;

    double rhs_sqnorm = rhs.squaredNorm();
    if (rhs_sqnorm == 0)
    {
        x->setZero();
        return true;
    }

   
    std::complex<double> alpha = 0;
    std::complex<double> w = 1;


    double tol2 = tol * tol * rhs_sqnorm;
    double eps2 = 1e-30;

    Eigen::VectorXcd x_p;
    x_p = *x;
    double relatedSolChange = 1e30;
    double eps = 1e-30;

    std::vector<Eigen::VectorXcd> us(L + 1);
    std::vector<Eigen::VectorXcd> rs(L + 1);
    Eigen::VectorXcd x0s(n);
    for (int i = 0; i <= L; i++) {
        us[i].resize(n);
        rs[i].resize(n);
    }

    Eigen::MatrixXcd tau(L + 1, L + 1);
    Eigen::VectorXcd sigma;
    sigma.resize(L + 1);

    Eigen::VectorXcd gamma;
    gamma.resize(L + 1);
    Eigen::VectorXcd gamma1;
    gamma1.resize(L + 1);
    Eigen::VectorXcd gamma2;
    gamma2.resize(L + 1);

    //Eigen::VectorXcd p = r;



    m_lastRelativeSolChange = std::pow((*x - x_p).squaredNorm() / (x_p.squaredNorm() + eps2), 0.5);
    m_error = std::pow((mat * (*x) - rhs).squaredNorm()/ rhs.squaredNorm(), 0.5);
    m_iters = 0;

    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(n), p = Eigen::VectorXcd::Zero(n), u = Eigen::VectorXcd::Zero(n);
    Eigen::VectorXcd y = Eigen::VectorXcd::Zero(n), z = Eigen::VectorXcd::Zero(n);

    Eigen::VectorXcd AKp = Eigen::VectorXcd::Zero(n);
    Eigen::VectorXcd AKs = Eigen::VectorXcd::Zero(n);

    Eigen::VectorXcd Ap = Eigen::VectorXcd::Zero(n);
    Eigen::VectorXcd precondr = Eigen::VectorXcd::Zero(n);
    Eigen::VectorXcd r_p = Eigen::VectorXcd::Zero(n);


    std::complex<double> beta = 0.0;
    Eigen::VectorXcd matKpr(n);
    Eigen::VectorXcd matu = Eigen::VectorXcd::Zero(n);
    std::complex<double> rho = dot(r0star, r);
    std::complex<double> rho_old=rho;
    

    m_numIterCorrection = 10;
    while (m_lastRelativeSolChange == 0.0 || (m_lastRelativeSolChange > tol && iter < maxIters)  || minIters > iter) {

        bool restart = false;
        
        //std::cout<<"divHds " << (divergenceCorrection->divergenceOperatorMatrix * (*x)).squaredNorm() << std::endl;
        if (abs(rho) < eps2 * r0_sqnorm)
        {
            // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
            // Let's restart with a new r0:
            r = rhs - mat *  (*x);
            r0star = r;
            rho = dot(r0star, r);

            restart = true;
            beta = 0.0;

        }

        precondr = precond.solve(r);
        matKpr = mat * precondr;
        
        p = precondr + beta * (p - u);
        Ap = matKpr + beta * (Ap - matu);
        alpha = rho / dot(r0star, Ap);


        std::complex<double> psi;
        std::complex<double> eta;
        if (iter ==0 || restart==true) { 
            psi = dot(matKpr, r) / dot(matKpr, matKpr);
            eta = 0.0;
        }
        else {
            std::complex<double> bb = dot(y, y);
            std::complex<double> ca = dot(matKpr, r);
            std::complex<double> ba = dot(y, r);
            std::complex<double> cb = dot(matKpr, y);
            std::complex<double> cc = dot(matKpr, matKpr);
            std::complex<double> bc = std::conj(cb);

            psi = (bb* ca - ba * cb) /
                (cc * bb - bc * cb);

            eta = (cc * ba - bc * ca) /
                (cc * bb - bc * cb);

            //psi = (dot(b, b) * dot(c, a) - dot(b, a) * dot(c, b)) /
            //    (dot(c, c) * dot(b, b) - dot(b, c) * dot(c, b));

            //eta = (dot(c, c) * dot(b, a) - dot(b, c) * dot(c, a)) /
            //    (dot(c, c) * dot(b, b) - dot(b, c) * dot(c, b));
        }
        
        u = precond.solve(psi * Ap + eta * y) + (eta * beta) * u;
        z = psi * precondr + eta * z - alpha * u;
        
        matu = mat * u;
        y = psi * matKpr + eta * y - alpha * matu;
        *x = *x + alpha * p + z;

        r_p = r;

        std::complex<double> rho_old = rho;

        r = r - alpha * Ap - y;

        rho = dot(r0star, r);

        beta = alpha / psi * rho/ rho_old;

        m_lastRelativeSolChange = std::pow((*x - x_p).squaredNorm() / (x_p.squaredNorm() + eps2), 0.5);
       
        //if ( iter > m_minNumIterCorrection && divergenceCorrection != nullptr && iter % m_numIterCorrection == 0) {
        //    Eigen::VectorXcd res = divergenceCorrection->correction(*x);
        //    std::cout << "before" << (divergenceCorrection->divergenceOperatorMatrix * (*x)).squaredNorm()<<" "<< (mat * (*x) - rhs).squaredNorm() << std::endl;
        //    std::cout << "xNorm " << x->squaredNorm() << " resNorm " << res.squaredNorm() << std::endl;
        //    for (int i = 0; i < n; i++) {
        //        std::cout << x->coeff(i) << " " << res.coeff(i) << std::endl;
        //    }
        //    *x += res;
        //    std::cout << "after" << (divergenceCorrection->divergenceOperatorMatrix * (*x)).squaredNorm() << " " << (mat * (*x) - rhs).squaredNorm() << std::endl;
        //    restart = true;
        //    //std::cout << std::pow((*x - x_p).squaredNorm() / (x_p.squaredNorm() + eps2), 0.5) << endl;
        //    r = rhs - mat * precond.solve(*x);
        //    rho = dot(r0star, r);

        //    beta = alpha / psi * rho / rho_old;
        //}
        //else {
        //    r = r - alpha * Ap - y;

        //    rho = dot(r0star, r);

        //    beta = alpha / psi * rho / rho_old;
        //}
        //std::cout<< std::pow(r.squaredNorm() / rhs.squaredNorm(), 0.5)<<" " << std::sqrt((*mat_in * (*x) - *rhs_in).squaredNorm() / rhs_in->squaredNorm()) << " " << m_lastRelativeSolChange << std::endl;

        x_p = *x;
        iter++;
    }
    m_error = std::pow(r.squaredNorm() / rhs.squaredNorm(), 0.5);

    m_iters = iter;

    if (iter == maxIters) {
        return false;
    }
    else {
        return true;
    }
    //return;
}
//
//void BiCGSafe::BiCGSafe::calculation(Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* mat, Eigen::VectorXcd* x, Eigen::VectorXcd* b, double tol, bool isDividedByDiagonal) {
//    if (!isDividedByDiagonal) {
//        solve(mat, x, b, tol);
//    }
//    else {
//        solveWithCorrection(mat, x, b, tol, m_numIterCorrection);
//    }
//}
//void BiCGSafe::BiCGSafe::solveWithCorrection(Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* mat, Eigen::VectorXcd* x, Eigen::VectorXcd* b, double tol, int numIteration) {
//    int keepMaxIteration = m_maxIteration;
//
//    bool isConverged;
//    m_maxIteration = m_minNumIterCorrection;
//    isConverged = solve(mat, x, b, tol);
//    if (!isConverged){
//        int iter = m_minNumIterCorrection;
//        m_numIterCorrection = 2000;
//        m_maxIteration = m_numIterCorrection;
//        while (iter < keepMaxIteration && !isConverged) {
//            Eigen::VectorXcd res = divergenceCorrection->correction(*x);
//            std::cout <<"before "<< std::sqrt((*mat * (*x) - *b).squaredNorm() / b->squaredNorm()) << std::endl;
//            *x += res;
//            std::cout <<"after "<< std::sqrt((*mat * (*x) - *b).squaredNorm() / b->squaredNorm()) << std::endl;
//            isConverged = solve(mat, x, b, tol);
//            iter += m_numIterCorrection;
//        }
//    }
//    m_maxIteration = keepMaxIteration;
//}

bool BiCGSafe::BiCGSafe::solve(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* mat, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x, const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* rhs, double tol, bool reuseILUByAdjoint) {
    //https://www.jstage.jst.go.jp/article/jsces/2005/0/2005_0_20050028/_pdf
    //BiCGSafe method based on minimization of associate residual
    //ê¸å`ï˚íˆéÆÇÃîΩïúâñ@ ä€ëP
    
    ofstream outputfile;
    outputfile.open("solverLog_" + to_string(omega) + ".txt", std::ios::out);
    outputfile << "relSolChange_1 relSolChange_2 error_1 error_2" << endl;

    bool isConverged = false;


    //precond.compute(*mat, reuseILUByAdjoint);
    precond.compute(*mat);


    int maxIters = m_maxIteration;

    int minIters = 10;


    int iter = 0;
    m_iters = iter;



    
    for (int i = 0; i < numOfSols; i++) {
        r0star[i].setZero();
        x0star[i].setZero();
    }


    const int seed = 12345;
    std::srand(seed);
    double randMax = RAND_MAX;
    for (int i = 0; i < numOfSols; i++) {
        for (int j = 0; j < r0star[i].size(); j++) {
            r0star[i].coeffRef(j) = std::rand() / randMax;
            r0star[i].coeffRef(j).imag(std::rand() / randMax);
        }
    }

    vector<Eigen::VectorXcd> r0norm(numOfSols);
    for (int i = 0; i < numOfSols; i++) {
        r[i].setZero();
        r[i] = (*rhs)[i] - *mat * ((*x)[i]);

        r0norm[i] = r[i];

        r0star[i] = r[i];
        r0_sqnorm[i] = r[i].squaredNorm();

        x0[i].setZero();
        x0[i] = (*x)[i];

        rhs_sqnorm[i] = (*rhs)[i].squaredNorm();
        if (rhs_sqnorm[i] == 0)
        {
            (*x)[i].setZero();

        }

        x_p[i].setZero();
        x_p[i] = (*x)[i];
    }


    for (int i = 0; i < numOfSols; i++) {
        alpha[i] = 0.0;
        w[i] = 1.0;
    }

    double eps2 = 1e-30;


    
    for (int i = 0; i < numOfSols; i++) {
        relatedSolChange[i] = 1e30;
    }
    double eps = 1e-30;


    //Eigen::VectorXcd p = r;

    vector<double> errorVector0;
    errorVector0.resize(numOfSols);
    
    for (int i = 0; i < numOfSols; i++) {
        m_lastRelativeSolChangeVector[i]= std::pow(((*x)[i] - x_p[i]).squaredNorm() / (x_p[i].squaredNorm() + eps2), 0.5);
        m_errorVector[i] = std::pow((*mat * (*x)[i] - (*rhs)[i]).squaredNorm() / (*rhs)[i].squaredNorm(), 0.5);
        errorVector0[i] = m_errorVector[i];
    }
    m_iters = 0;

    

    for (int i = 0; i < numOfSols; i++) {
        v[i].setZero();

        p[i].setZero();

        u[i].setZero();

        y[i].setZero();

        z[i].setZero();

        Ap[i].setZero();

        precondr[i].setZero();

        r_p[i].setZero();

        matKpr[i].setZero();

        matu[i].setZero();

        psiApPlusEtaY[i].setZero();
    }

    
    for (int i = 0; i < numOfSols; i++) {
        beta[i] = 0.0;
    }


    
    for (int i = 0; i < numOfSols; i++) {
        rho[i] = r0star[i].dot(r[i]);
        rho_old[i] = rho[i];
    }

    
    std::complex<double> bb ;
    std::complex<double> ca ;
    std::complex<double> ba ;
    std::complex<double> cb ;
    std::complex<double> cc ;
    std::complex<double> bc ;

    vector<bool>finishedEachSols(numOfSols);
    for (int i = 0; i < numOfSols; i++) {
        finishedEachSols[i] = false;
    }

    while(true){
    //while (m_lastRelativeSolChange == 0.0 || (m_lastRelativeSolChange > tol && iter < maxIters) || minIters > iter) {
        //cout<<iter<<" " << m_errorVector[0] << " " << m_errorVector[1] << " " << m_lastRelativeSolChangeVector[0] << " " << m_lastRelativeSolChangeVector[1] << endl;

        outputfile << m_lastRelativeSolChangeVector[0] << " " << m_lastRelativeSolChangeVector[1] << " " << m_errorVector[0] << " " << m_errorVector[1] << endl;
        bool isFinite = true;
        for (int i = 0; i < numOfSols; i++) {
            bool tmp=isfinite(m_errorVector[i]);
            isFinite = isFinite * tmp;
            tmp = isfinite(m_lastRelativeSolChangeVector[i]);
            isFinite = isFinite * tmp;
        }

        if (!isFinite) {
            m_iters = maxIters;
            iter = maxIters;
            break;
        }

        m_lastRelativeSolChange = 0.0;
        for (int i = 0; i < numOfSols; i++) {
            if (m_lastRelativeSolChangeVector[i] > m_lastRelativeSolChange) {
                m_lastRelativeSolChange = m_lastRelativeSolChangeVector[i];
            }
        }
        

        if (iter != 0) {
            for (int i = 0; i < numOfSols; i++) {
                if(!(m_lastRelativeSolChangeVector[i] > tol && iter < maxIters)) {
                    finishedEachSols[i] = true;
                }
            }
        }

        for (int i = 0; i < numOfSols; i++) {
            if (!(m_errorVector[i] > tol && iter < maxIters)) {
                finishedEachSols[i] = true;
            }
        }

        bool allFinished = true;
        for (int i = 0; i < numOfSols; i++) {
            if (finishedEachSols[i] ==false) {
                allFinished = false;
            }
        }

        if (allFinished) {
            break;
        }

        
        for (int i = 0; i < numOfSols; i++) {
            restart[i] = false;
        }

        //std::cout<<"divHds " << (divergenceCorrection->divergenceOperatorMatrix * (*x)).squaredNorm() << std::endl;
        for (int i = 0; i < numOfSols; i++) {
            if (finishedEachSols[i]) {
                continue;
            }
            if (abs(rho[i]) < eps2 * r0_sqnorm[i])
            {
                // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
                // Let's restart with a new r0:
                r[i] = (*rhs)[i] - *mat *(*x)[i];
                r0star[i] = r[i];
                rho[i] = r0star[i].dot(r[i]);

                restart[i] = true;
                beta[i] = 0.0;

            }
        }
        for (int i = 0; i < numOfSols; i++) {
            if (finishedEachSols[i]) {
                continue;
            }
             precond.solve(r[i], precondr[i]);

            matKpr[i] = *mat * precondr[i];
            //precondr[i] = precond.solve(r[i]);
            //matKpr[i] = *mat * precondr[i];
        }
       
        for (int i = 0; i < numOfSols; i++) {
            if (finishedEachSols[i]) {
                continue;
            }
            p[i] = precondr[i] + beta[i] * (p[i] - u[i]);
            Ap[i] = matKpr[i] + beta[i] * (Ap[i] - matu[i]);
            alpha[i] = rho[i] / r0star[i].dot(Ap[i]);
        }
        


        
        for (int i = 0; i < numOfSols; i++) {
            if (finishedEachSols[i]) {
                continue;
            }
            if (iter == 0 || restart[i] == true) {
                psi[i] = matKpr[i].dot(r[i]) / matKpr[i].dot(matKpr[i]);
                eta[i] = 0.0;
            }
            else {
                bb = y[i].dot(y[i]);
                ca = matKpr[i].dot(r[i]);
                ba = y[i].dot(r[i]);
                cb = matKpr[i].dot(y[i]);
                cc = matKpr[i].dot(matKpr[i]);
                bc = std::conj(cb);

                psi[i] = (bb * ca - ba * cb) /
                    (cc * bb - bc * cb);

                eta[i] = (cc * ba - bc * ca) /
                    (cc * bb - bc * cb);

            }

            psiApPlusEtaY[i] = psi[i] * Ap[i] + eta[i] * y[i];
            Eigen::VectorXcd tmp{ mat->rows() };
            precond.solve(psiApPlusEtaY[i], tmp);
            
            u[i] = tmp + (eta[i] * beta[i]) * u[i];
            z[i] = psi[i] * precondr[i] + eta[i] * z[i] - alpha[i] * u[i];


    
            matu[i] = *mat * u[i];


            y[i] = psi[i] * matKpr[i] + eta[i] * y[i] - alpha[i] * matu[i];
            (*x)[i] = (*x)[i] + alpha[i] * p[i] + z[i];

            r_p[i] = r[i];

            rho_old[i] = rho[i];

            r[i] = r[i] - alpha[i] * Ap[i] - y[i];

            rho[i] = r0star[i].dot( r[i]);

            beta[i] = alpha[i] / psi[i] * rho[i] / rho_old[i];

            m_lastRelativeSolChangeVector[i] = std::pow(((*x)[i] - x_p[i]).squaredNorm() / (x_p[i].squaredNorm() + eps2), 0.5);
            x_p[i] = (*x)[i];

            m_errorVector[i] = std::pow(r[i].squaredNorm() / rhs_sqnorm[i], 0.5)/ errorVector0[i];
            //m_errorVector[i] = std::pow((*mat * (*x)[i] - (*rhs)[i]).squaredNorm() / (*rhs)[i].squaredNorm(), 0.5);

            double rmax = 0;
            int argRmax = 0;
            for (int j = 0; j < r[i].size(); j++) {
                if (rmax < abs(r[i].coeff(j)/r0norm[i].coeff(j)) && r0norm[i].coeff(j) != 0.0) {
                    rmax = abs(r[i].coeff(j) / r0norm[i].coeff(j));
                    argRmax = j;
                }
            }
            /*Element::Element* element = (*calcElementsVector)[(*solverToOriginal)[argRmax]/3];
            cout << "Direc,RMax,elementID,XYZ,Resistivity,preR:" << i << " " << rmax << " " << element->ID<<" "<< (*solverToOriginal)[argRmax] % 3 << " " << element->resistivity
            << " " << abs(r0norm[i].coeff(argRmax)) << endl;
            double rmin = 1e30;
            int argRmin = 0;
            for (int j = 0; j < r[i].size(); j++) {
                if (rmin > abs(r[i].coeff(j) / r0norm[i].coeff(j)) && r0norm[i].coeff(j)!=0.0) {
                    rmin = abs(r[i].coeff(j) / r0norm[i].coeff(j));
                    argRmin = j;
                }
            }
            element = (*calcElementsVector)[(*solverToOriginal)[argRmin] / 3];
            cout << "Direc,RMin,elementID,XYZ,Resistivity,preR:" << i << " " << rmin << " " << element->ID << " " << (*solverToOriginal)[argRmin] % 3 << " " << element->resistivity
                <<" "<<abs(r0norm[i].coeff(argRmin)) << endl;

            r0norm[i] = r[i];*/

        }
        

        iter++;
    }
    m_error = 0.0;

    for (int i = 0; i < numOfSols; i++) {
        if (std::pow(r[i].squaredNorm() / (*rhs)[i].squaredNorm(), 0.5) > m_error) {
            m_error = std::pow(r[i].squaredNorm() / (*rhs)[i].squaredNorm(), 0.5);
        }
        rReturn[i] = r[i];
        //(*x)[i] = precond.RecoverSolution((*x)[i]);
    }


    m_iters = iter;
    outputfile.close();

    if (iter == maxIters) {
        return false;
    }
    else {
        return true;
    }
    //return;
}
bool BiCGSafe::BiCGSafe::solve(const Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor>* mat, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x, const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* rhs, double tol, bool reuseILUByAdjoint) {
    //https://www.jstage.jst.go.jp/article/jsces/2005/0/2005_0_20050028/_pdf
    //BiCGSafe method based on minimization of associate residual
    //ê¸å`ï˚íˆéÆÇÃîΩïúâñ@ ä€ëP




    bool isConverged = false;

    //precond.compute(*mat,reuseILUByAdjoint);
    precond.compute(*mat);



    int maxIters = m_maxIteration;

    int minIters = 10;


    int iter = 0;
    m_iters = iter;




    for (int i = 0; i < numOfSols; i++) {
        r0star[i].setZero();
        x0star[i].setZero();
    }


    const int seed = 12345;
    std::srand(seed);
    double randMax = RAND_MAX;
    for (int i = 0; i < numOfSols; i++) {
        for (int j = 0; j < r0star[i].size(); j++) {
            r0star[i].coeffRef(j) = std::rand() / randMax;
            r0star[i].coeffRef(j).imag(std::rand() / randMax);
        }
    }


    for (int i = 0; i < numOfSols; i++) {
        r[i].setZero();
        r[i] = (*rhs)[i] - *mat * ((*x)[i]);

        r0star[i] = r[i];
        r0_sqnorm[i] = r[i].squaredNorm();

        x0[i].setZero();
        x0[i] = (*x)[i];

        rhs_sqnorm[i] = (*rhs)[i].squaredNorm();
        if (rhs_sqnorm[i] == 0)
        {
            (*x)[i].setZero();

        }

        x_p[i].setZero();
        x_p[i] = (*x)[i];
    }


    for (int i = 0; i < numOfSols; i++) {
        alpha[i] = 0.0;
        w[i] = 1.0;
    }

    double eps2 = 1e-30;



    for (int i = 0; i < numOfSols; i++) {
        relatedSolChange[i] = 1e30;
    }
    double eps = 1e-30;


    //Eigen::VectorXcd p = r;



    for (int i = 0; i < numOfSols; i++) {
        m_lastRelativeSolChangeVector[i] = std::pow(((*x)[i] - x_p[i]).squaredNorm() / (x_p[i].squaredNorm() + eps2), 0.5);

        m_errorVector[i] = std::pow((*mat * (*x)[i] - (*rhs)[i]).squaredNorm() / (*rhs)[i].squaredNorm(), 0.5);
    }
    m_iters = 0;



    for (int i = 0; i < numOfSols; i++) {
        v[i].setZero();

        p[i].setZero();

        u[i].setZero();

        y[i].setZero();

        z[i].setZero();

        Ap[i].setZero();

        precondr[i].setZero();

        r_p[i].setZero();

        matKpr[i].setZero();

        matu[i].setZero();

        psiApPlusEtaY[i].setZero();

    }


    for (int i = 0; i < numOfSols; i++) {
        beta[i] = 0.0;
    }



    for (int i = 0; i < numOfSols; i++) {
        rho[i] = r0star[i].dot(r[i]);
        rho_old[i] = rho[i];
    }


    vector<std::complex<double>> bb;
    vector < std::complex<double>> ca;
    vector < std::complex<double>> ba;
    vector < std::complex<double>> cb;
    vector < std::complex<double>> cc;
    vector < std::complex<double>> bc;
    bb.resize(numOfSols);
    ca.resize(numOfSols);
    ba.resize(numOfSols);
    cb.resize(numOfSols);
    cc.resize(numOfSols);
    bc.resize(numOfSols);

    while (true) {
        //while (m_lastRelativeSolChange == 0.0 || (m_lastRelativeSolChange > tol && iter < maxIters) || minIters > iter) {
        //cout << iter << " " << m_errorVector[0] << " " << m_errorVector[1] << " " << m_lastRelativeSolChangeVector[0] << " " << m_lastRelativeSolChangeVector[1] << endl;


        bool isFinite = true;
        for (int i = 0; i < numOfSols; i++) {
            bool tmp = isfinite(m_errorVector[i]);
            isFinite = isFinite * tmp;
            tmp = isfinite(m_lastRelativeSolChangeVector[i]);
            isFinite = isFinite * tmp;
        }

        if (!isFinite) {
            m_iters = maxIters;
            iter = maxIters;
            break;
        }

        m_lastRelativeSolChange = 0.0;
        for (int i = 0; i < numOfSols; i++) {
            if (m_lastRelativeSolChangeVector[i] > m_lastRelativeSolChange) {
                m_lastRelativeSolChange = m_lastRelativeSolChangeVector[i];
            }
        }

        bool continueCalc = false;
        if (iter == 0) {
            continueCalc = true;
        }
        for (int i = 0; i < numOfSols; i++) {
            if (m_lastRelativeSolChangeVector[i] > tol && iter < maxIters) {
                continueCalc = true;
            }

        }

        if (!continueCalc) {
            break;
        }

        continueCalc = false;
        for (int i = 0; i < numOfSols; i++) {
            if (m_errorVector[i] > tol && iter < maxIters) {
                continueCalc = true;
            }
        }

        if (!continueCalc) {
            break;
        }


        for (int i = 0; i < numOfSols; i++) {
            restart[i] = false;
        }

        //std::cout<<"divHds " << (divergenceCorrection->divergenceOperatorMatrix * (*x)).squaredNorm() << std::endl;
        for (int i = 0; i < numOfSols; i++) {
            if (abs(rho[i]) < eps2 * r0_sqnorm[i])
            {
                // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
                // Let's restart with a new r0:
                r[i] = (*rhs)[i] - *mat * (*x)[i];
                r0star[i] = r[i];
                rho[i] = r0star[i].dot(r[i]);

                restart[i] = true;
                beta[i] = 0.0;

            }
        }
        for (int i = 0; i < numOfSols; i++) {

            precond.solve(r[i], precondr[i]);

            matKpr[i] = *mat * precondr[i];

            //precondr[i] = precond.solve(r[i]);
            //matKpr[i] = *mat * precondr[i];
        }

        for (int i = 0; i < numOfSols; i++) {
            p[i] = precondr[i] + beta[i] * (p[i] - u[i]);
            Ap[i] = matKpr[i] + beta[i] * (Ap[i] - matu[i]);
            alpha[i] = rho[i] / r0star[i].dot(Ap[i]);
        }




        for (int i = 0; i < numOfSols; i++) {
            if (iter == 0 || restart[i] == true) {
                psi[i] = matKpr[i].dot(r[i]) / matKpr[i].dot(matKpr[i]);
                eta[i] = 0.0;
            }
            else {
                bb[i] = y[i].dot(y[i]);
                ca[i] = matKpr[i].dot(r[i]);
                ba[i] = y[i].dot(r[i]);
                cb[i] = matKpr[i].dot(y[i]);
                cc[i] = matKpr[i].dot(matKpr[i]);
                bc[i] = std::conj(cb[i]);

                psi[i] = (bb[i] * ca[i] - ba[i] * cb[i]) /
                    (cc[i] * bb[i] - bc[i] * cb[i]);

                eta[i] = (cc[i] * ba[i] - bc[i] * ca[i]) /
                    (cc[i] * bb[i] - bc[i] * cb[i]);

            }

            psiApPlusEtaY[i] = psi[i] * Ap[i] + eta[i] * y[i];
            Eigen::VectorXcd tmp{ mat->rows() };
            precond.solve(psiApPlusEtaY[i], tmp);
            u[i] = tmp + (eta[i] * beta[i]) * u[i];
            z[i] = psi[i] * precondr[i] + eta[i] * z[i] - alpha[i] * u[i];



            matu[i] = *mat * u[i];


            y[i] = psi[i] * matKpr[i] + eta[i] * y[i] - alpha[i] * matu[i];
            (*x)[i] = (*x)[i] + alpha[i] * p[i] + z[i];

            r_p[i] = r[i];

            rho_old[i] = rho[i];

            r[i] = r[i] - alpha[i] * Ap[i] - y[i];

            rho[i] = r0star[i].dot(r[i]);

            beta[i] = alpha[i] / psi[i] * rho[i] / rho_old[i];

            m_lastRelativeSolChangeVector[i] = std::pow(((*x)[i] - x_p[i]).squaredNorm() / (x_p[i].squaredNorm() + eps2), 0.5);
            x_p[i] = (*x)[i];


            m_errorVector[i] = std::pow(r[i].squaredNorm() / rhs_sqnorm[i], 0.5);


        }

        iter++;
    }
    m_error = 0.0;
    
    for (int i = 0; i < numOfSols; i++) {
        if (std::pow(r[i].squaredNorm() / (*rhs)[i].squaredNorm(), 0.5) > m_error) {
            m_error = std::pow(r[i].squaredNorm() / (*rhs)[i].squaredNorm(), 0.5);
        }
        rReturn[i] = r[i];

    }


    m_iters = iter;

    if (iter == maxIters) {
        return false;
    }
    else {
        return true;
    }
    //return;
}

bool BiCGSafe::BiCGSafe::solveWithSSOR(const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* mat, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x, const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* rhs_in, double tol, bool isDividedByDiagonal) {
    ////https://www.jstage.jst.go.jp/article/jsces/2005/0/2005_0_20050028/_pdf
    ////BiCGSafe method based on minimization of associate residual
    ////ê¸å`ï˚íˆéÆÇÃîΩïúâñ@ ä€ëP


    //vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> rhs(rhs_in->size());
    //vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> x_init(rhs_in->size());
    //for (int i = 0; i < rhs_in->size(); i++) {
    //    rhs[i].resize(mat->cols());
    //    x_init[i].resize(mat->cols());
    //    for (int j = 0; j < rhs[i].size(); j++) {
    //        rhs[i].coeffRef(j) = (*rhs_in)[i].coeff(j);
    //        x_init[i].coeffRef(j) = (*x)[i].coeff(j);
    //    }
    //}
    //

    //bool isConverged = false;

    ////Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> mat{ n,n };
    ////mat.reserve(Eigen::VectorXi::Constant(n, 100));
    ////Eigen::MatrixXcd rhs(n, numOfSols);
    ////for (int i = 0; i < n; i++) {
    ////    for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*mat_in, i); it; ++it) {
    ////        mat.coeffRef(i, it.col()) = mat_in->coeff(i, it.col());
    ////    }

    ////}
    ////mat.eval();
    ////std::cout << "b" << std::endl;
    ////
    ////rhs = *rhs_in;
    ////std::cout << "c" << std::endl;




    ////ILU::ILU precond;
    ////Eigen::DiagonalPreconditioner<std::complex<double>> precond;
    ////precond.compute(*mat);


    //int maxIters = m_maxIteration;

    //int minIters = 10;


    //int iter = 0;





    //for (int i = 0; i < numOfSols; i++) {
    //    r0star[i].setZero();
    //    x0star[i].setZero();
    //}


    //const int seed = 12345;
    //std::srand(seed);
    //double randMax = RAND_MAX;
    //for (int i = 0; i < numOfSols; i++) {
    //    for (int j = 0; j < r0star[i].size(); j++) {
    //        r0star[i].coeffRef(j) = std::rand() / randMax;
    //        r0star[i].coeffRef(j).imag(std::rand() / randMax);
    //    }
    //}

    //EisenstatSSOR::EisenstatSSOR precond;
    //precond.compute(mat);

    //for (int i = 0; i < numOfSols; i++) {
    //    r[i].setZero();
    //    r[i] = rhs[i] - *mat * ((*x)[i]);

    //    
    //    precond.ChangeVectors((*x)[i], rhs[i], r[i]);

    //    //r0star[i] = r[i];
    //    r0_sqnorm[i] = r[i].squaredNorm();

    //    x0[i].setZero();
    //    x0[i] = (*x)[i];

    //    rhs_sqnorm[i] = rhs[i].squaredNorm();
    //    if (rhs_sqnorm[i] == 0)
    //    {
    //        (*x)[i].setZero();

    //    }
    //    
    //    x_p[i].setZero();
    //    x_p[i] = (*x)[i];
    //}


    //for (int i = 0; i < numOfSols; i++) {
    //    alpha[i] = 0.0;
    //    w[i] = 1.0;
    //}

    //double eps2 = 1e-30;



    //for (int i = 0; i < numOfSols; i++) {
    //    relatedSolChange[i] = 1e30;
    //}
    //double eps = 1e-30;


    ////Eigen::VectorXcd p = r;



    //for (int i = 0; i < numOfSols; i++) {

    //    m_lastRelativeSolChangeVector[i] = std::pow(((*x)[i] - x_p[i]).squaredNorm() / (x_p[i].squaredNorm() + eps2), 0.5);
    //    m_errorVector[i] = std::pow(( precond.Dot ((*x)[i]) - rhs[i]).squaredNorm() / rhs[i].squaredNorm(), 0.5);
    //}
    //m_iters = 0;



    //for (int i = 0; i < numOfSols; i++) {
    //    v[i].setZero();

    //    p[i].setZero();

    //    u[i].setZero();

    //    y[i].setZero();

    //    z[i].setZero();

    //    Ap[i].setZero();

    //    precondr[i].setZero();

    //    r_p[i].setZero();

    //    matKpr[i].setZero();

    //    matu[i].setZero();

    //}


    //for (int i = 0; i < numOfSols; i++) {
    //    beta[i] = 0.0;
    //}



    //for (int i = 0; i < numOfSols; i++) {
    //    rho[i] = r0star[i].dot(r[i]);
    //    rho_old[i] = rho[i];
    //}

    //cout << (precond.Dot((*x)[0]) - rhs[0]).squaredNorm() << endl;


    //std::complex<double> bb;
    //std::complex<double> ca;
    //std::complex<double> ba;
    //std::complex<double> cb;
    //std::complex<double> cc;
    //std::complex<double> bc;

    //while (true) {

    //    //while (m_lastRelativeSolChange == 0.0 || (m_lastRelativeSolChange > tol && iter < maxIters) || minIters > iter) {

    //    m_lastRelativeSolChange = 0.0;
    //    for (int i = 0; i < numOfSols; i++) {
    //        if (m_lastRelativeSolChangeVector[i] > m_lastRelativeSolChange) {
    //            m_lastRelativeSolChange = m_lastRelativeSolChangeVector[i];
    //        }
    //    }

    //    /*bool continueCalc = false;
    //    if (iter == 0) {
    //        continueCalc = true;
    //    }
    //    for (int i = 0; i < numOfSols; i++) {
    //        if (m_lastRelativeSolChangeVector[i] > tol && iter < maxIters) {
    //            continueCalc = true;
    //        }

    //    }

    //    if (!continueCalc) {
    //        break;
    //    }*/



    //    //continueCalc = false;
    //    //for (int i = 0; i < numOfSols; i++) {
    //    //    if (m_errorVector[i] > tol && iter < maxIters) {
    //    //        continueCalc = true;
    //    //    }
    //    //}

    //    //if (!continueCalc) {
    //    //    break;
    //    //}


    //    for (int i = 0; i < numOfSols; i++) {
    //        restart[i] = false;
    //    }

    //    //std::cout<<"divHds " << (divergenceCorrection->divergenceOperatorMatrix * (*x)).squaredNorm() << std::endl;
    //    for (int i = 0; i < numOfSols; i++) {
    //        if (abs(rho[i]) < eps2 * r0_sqnorm[i])
    //        {
    //            // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
    //            // Let's restart with a new r0:
    //            r[i] = rhs[i] - precond.Dot( (*x)[i]);
    //            r0star[i] = r[i];
    //            rho[i] = r0star[i].dot(r[i]);

    //            restart[i] = true;
    //            beta[i] = 0.0;
    //            cout << "restart" << endl;
    //        }
    //    }
    //    for (int i = 0; i < numOfSols; i++) {
    //        matKpr[i] = precond.Dot(r[i]);
    //    }

    //    for (int i = 0; i < numOfSols; i++) {
    //        p[i] = r[i] + beta[i] * (p[i] - u[i]);
    //        Ap[i] = matKpr[i] + beta[i] * (Ap[i] - matu[i]);
    //        cout << matKpr[i].squaredNorm()<<" "<<beta[i] << " " << matu[i].squaredNorm() << endl;
    //        alpha[i] = rho[i] / r0star[i].dot(Ap[i]);
    //        
    //    }




    //    for (int i = 0; i < numOfSols; i++) {
    //        if (iter == 0 || restart[i] == true) {
    //            psi[i] = matKpr[i].dot(r[i]) / matKpr[i].dot(matKpr[i]);
    //            eta[i] = 0.0;
    //        }
    //        else {
    //            bb = y[i].dot(y[i]);
    //            ca = matKpr[i].dot(r[i]);
    //            ba = y[i].dot(r[i]);
    //            cb = matKpr[i].dot(y[i]);
    //            cc = matKpr[i].dot(matKpr[i]);
    //            bc = std::conj(cb);

    //            psi[i] = (bb * ca - ba * cb) /
    //                (cc * bb - bc * cb);

    //            eta[i] = (cc * ba - bc * ca) /
    //                (cc * bb - bc * cb);

    //        }
    //        u[i] = psi[i] * Ap[i] + eta[i] * y[i] + (eta[i] * beta[i]) * u[i];
    //        z[i] = psi[i] * r[i] + eta[i] * z[i] - alpha[i] * u[i];
    //        


    //        matu[i] = precond.Dot(u[i]);


    //        y[i] = psi[i] * matKpr[i] + eta[i] * y[i] - alpha[i] * matu[i];
    //        (*x)[i] = (*x)[i] + alpha[i] * p[i] + z[i];

    //        r_p[i] = r[i];

    //        rho_old[i] = rho[i];
    //        r[i] = r[i] - alpha[i] * Ap[i] - y[i];
    //        
    //        rho[i] = r0star[i].dot(r[i]);

    //        beta[i] = alpha[i] / psi[i] * rho[i] / rho_old[i];

    //        m_lastRelativeSolChangeVector[i] = std::pow(((*x)[i] - x_p[i]).squaredNorm() / (x_p[i].squaredNorm() + eps2), 0.5);
    //        x_p[i] = (*x)[i];
    //        

    //        m_errorVector[i] = std::pow(r[i].squaredNorm() / rhs_sqnorm[i], 0.5);

    //        Eigen::VectorXcd tmpx=(*x)[i];
    //        precond.RecoverSolution(tmpx);
    //        //cout << (*mat* tmpx - (*rhs_in)[i]).squaredNorm() / (*rhs_in)[i].squaredNorm() << endl;

    //    }

    //    iter++;
    //}

    //for (int i = 0; i < numOfSols; i++) {
    //    precond.RecoverSolution((*x)[i]);
    //}
    //m_error = 0.0;

    //for (int i = 0; i < numOfSols; i++) {
    //    if (std::pow(r[i].squaredNorm() / rhs[i].squaredNorm(), 0.5) > m_error) {
    //        m_error = std::pow(r[i].squaredNorm() / rhs[i].squaredNorm(), 0.5);
    //    }

    //}


    //m_iters = iter;

    //if (iter == maxIters) {
    //    return false;
    //}
    //else {
    //    return true;
    //}
    return false;
}




bool BiCGSafe::BiCGSafe::solve(const Eigen::SparseMatrix<double, Eigen::RowMajor>* matR, const Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* matI, vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* x, const vector<Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>>* rhs, double tol, bool reuseILUByAdjoint) {
    //https://www.jstage.jst.go.jp/article/jsces/2005/0/2005_0_20050028/_pdf
    //BiCGSafe method based on minimization of associate residual
    //ê¸å`ï˚íˆéÆÇÃîΩïúâñ@ ä€ëP

    ofstream outputfile;
    outputfile.open("solverLog_" + to_string(omega) + ".txt", std::ios::out);
    outputfile << "numOfIterations polarization relSolChange_1 relSolChange_2 error_1 error_2" << endl;


    bool isConverged = false;


    int maxIters = m_maxIteration;

    int minIters = 10;


    //int iter = 0;
    vector<int> iters(numOfSols);
    for (int i = 0; i < numOfSols; i++) {
        iters[i] = 0;
    }
    m_iters = 0;




    for (int i = 0; i < numOfSols; i++) {
        r0star[i].setZero();
        x0star[i].setZero();
    }


    /*const int seed = 12345;
    std::srand(seed);
    double randMax = RAND_MAX;
    for (int i = 0; i < numOfSols; i++) {
        for (int j = 0; j < r0star[i].size(); j++) {
            r0star[i].coeffRef(j) = std::rand() / randMax;
            r0star[i].coeffRef(j).imag(std::rand() / randMax);
        }
    }*/

    vector<Eigen::VectorXcd> r0norm(numOfSols);
    for (int i = 0; i < numOfSols; i++) {
        r[i].setZero();
        r[i] = (*rhs)[i] - *matR * ((*x)[i]) - *matI * ((*x)[i]);

        r0norm[i] = r[i];

        r0star[i] = r[i];
        r0_sqnorm[i] = r[i].squaredNorm();

        x0[i].setZero();
        x0[i] = (*x)[i];

        rhs_sqnorm[i] = (*rhs)[i].squaredNorm();
        if (rhs_sqnorm[i] == 0)
        {
            (*x)[i].setZero();

        }

        x_p[i].setZero();
        x_p[i] = (*x)[i];
    }


    for (int i = 0; i < numOfSols; i++) {
        alpha[i] = 0.0;
        w[i] = 1.0;
    }

    double eps2 = 1e-30;



    for (int i = 0; i < numOfSols; i++) {
        relatedSolChange[i] = 1e30;
    }
    double eps = 1e-30;


    //Eigen::VectorXcd p = r;



    for (int i = 0; i < numOfSols; i++) {
        m_lastRelativeSolChangeVector[i] = std::pow(((*x)[i] - x_p[i]).squaredNorm() / (x_p[i].squaredNorm() + eps2), 0.5);
        m_errorVector[i] = std::pow((*matR * (*x)[i] + *matI * (*x)[i] - (*rhs)[i]).squaredNorm() / (*rhs)[i].squaredNorm(), 0.5);
    }
    m_iters = 0;



    for (int i = 0; i < numOfSols; i++) {
        v[i].setZero();

        p[i].setZero();

        u[i].setZero();

        y[i].setZero();

        z[i].setZero();

        Ap[i].setZero();

        precondr[i].setZero();

        r_p[i].setZero();

        matKpr[i].setZero();

        matu[i].setZero();

        psiApPlusEtaY[i].setZero();
    }


    for (int i = 0; i < numOfSols; i++) {
        beta[i] = 0.0;
    }



    for (int i = 0; i < numOfSols; i++) {
        rho[i] = r0star[i].dot(r[i]);
        rho_old[i] = rho[i];
    }


 

    vector<bool>finishedEachSols(numOfSols);
    for (int i = 0; i < numOfSols; i++) {
        finishedEachSols[i] = false;
    }
#pragma omp parallel for
    for (int i = 0; i < numOfSols; i++) {
        while (true) {
            //while (m_lastRelativeSolChange == 0.0 || (m_lastRelativeSolChange > tol && iter < maxIters) || minIters > iter) {
            //cout<<iter<<" " << m_errorVector[0] << " " << m_errorVector[1] << " " << m_lastRelativeSolChangeVector[0] << " " << m_lastRelativeSolChangeVector[1] << endl;

            outputfile<<iters[i]<<" " << i  << " " << m_lastRelativeSolChangeVector[i] << " " << m_errorVector[i] << endl;



            bool isFinite = true;
            bool chkfinite = isfinite(m_errorVector[i]);
            isFinite = isFinite * chkfinite;
            chkfinite = isfinite(m_lastRelativeSolChangeVector[i]);
            isFinite = isFinite * chkfinite;


            if (!isFinite) {
                m_iters = maxIters;
                break;
            }

            



            if (iters[i] != 0) {
                if (!(m_lastRelativeSolChangeVector[i] > tol && iters[i] < maxIters)) {
                    finishedEachSols[i] = true;   
                }
            }

            if (!(m_errorVector[i] > tol && iters[i] < maxIters)) {
                finishedEachSols[i] = true;
            }


            restart[i] = false;



            if (finishedEachSols[i]) {
                break;
            }
            if (abs(rho[i]) < eps2 * r0_sqnorm[i])
            {
                // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
                // Let's restart with a new r0:
                r[i] = (*rhs)[i] - *matR * (*x)[i] - *matI * (*x)[i];
                r0star[i] = r[i];
                rho[i] = r0star[i].dot(r[i]);

                restart[i] = true;
                beta[i] = 0.0;

            }

            if (finishedEachSols[i]) {
                continue;
            }
            precond.solve(r[i], precondr[i]);
            //if (useMultiGrid) {
            //    precond_multi.solve(r[i], precondr[i]);
            //}
            //else {
            //    precond.solve(r[i], precondr[i]);
            //}

            matKpr[i] = *matR * precondr[i] + *matI * precondr[i];

            //precondr[i] = precond.solve(r[i]);
            //matKpr[i] = *mat * precondr[i];

            if (finishedEachSols[i]) {
                continue;
            }
            p[i] = precondr[i] + beta[i] * (p[i] - u[i]);
            Ap[i] = matKpr[i] + beta[i] * (Ap[i] - matu[i]);
            alpha[i] = rho[i] / r0star[i].dot(Ap[i]);

            if (finishedEachSols[i]) {
                continue;
            }
            if (iters[i] == 0 || restart[i] == true) {
                psi[i] = matKpr[i].dot(r[i]) / matKpr[i].dot(matKpr[i]);
                eta[i] = 0.0;
            }
            
            else {
                std::complex<double> bb;
                std::complex<double> ca;
                std::complex<double> ba;
                std::complex<double> cb;
                std::complex<double> cc;
                std::complex<double> bc;

                bb = y[i].dot(y[i]);
                ca = matKpr[i].dot(r[i]);
                ba = y[i].dot(r[i]);
                cb = matKpr[i].dot(y[i]);
                cc = matKpr[i].dot(matKpr[i]);
                bc = std::conj(cb);

                psi[i] = (bb * ca - ba * cb) /
                    (cc * bb - bc * cb);

                eta[i] = (cc * ba - bc * ca) /
                    (cc * bb - bc * cb);

            }
            
            psiApPlusEtaY[i] = psi[i] * Ap[i] + eta[i] * y[i];
            
            Eigen::VectorXcd tmp{ matR->rows() };
            precond.solve(psiApPlusEtaY[i], tmp);

            /*if (useMultiGrid) {
                precond_multi.solve(psiApPlusEtaY[i], tmp);
            }
            else {
                precond.solve(psiApPlusEtaY[i], tmp);
            }*/
            u[i] = tmp + (eta[i] * beta[i]) * u[i];
            z[i] = psi[i] * precondr[i] + eta[i] * z[i] - alpha[i] * u[i];



            matu[i] = *matR * u[i] + *matI * u[i];


            y[i] = psi[i] * matKpr[i] + eta[i] * y[i] - alpha[i] * matu[i];
            (*x)[i] = (*x)[i] + alpha[i] * p[i] + z[i];

            r_p[i] = r[i];

            rho_old[i] = rho[i];

            r[i] = r[i] - alpha[i] * Ap[i] - y[i];

            rho[i] = r0star[i].dot(r[i]);

            beta[i] = alpha[i] / psi[i] * rho[i] / rho_old[i];

            m_lastRelativeSolChangeVector[i] = std::pow(((*x)[i] - x_p[i]).squaredNorm() / (x_p[i].squaredNorm() + eps2), 0.5);
            x_p[i] = (*x)[i];

            m_errorVector[i] = std::pow(r[i].squaredNorm() / rhs_sqnorm[i], 0.5);

            double rmax = 0;
            int argRmax = 0;
            for (int j = 0; j < r[i].size(); j++) {
                if (rmax < abs(r[i].coeff(j) / r0norm[i].coeff(j)) && r0norm[i].coeff(j) != 0.0) {
                    rmax = abs(r[i].coeff(j) / r0norm[i].coeff(j));
                    argRmax = j;
                }
            }
            /*Element::Element* element = (*calcElementsVector)[(*solverToOriginal)[argRmax]/3];
            cout << "Direc,RMax,elementID,XYZ,Resistivity,preR:" << i << " " << rmax << " " << element->ID<<" "<< (*solverToOriginal)[argRmax] % 3 << " " << element->resistivity
            << " " << abs(r0norm[i].coeff(argRmax)) << endl;
            double rmin = 1e30;
            int argRmin = 0;
            for (int j = 0; j < r[i].size(); j++) {
                if (rmin > abs(r[i].coeff(j) / r0norm[i].coeff(j)) && r0norm[i].coeff(j)!=0.0) {
                    rmin = abs(r[i].coeff(j) / r0norm[i].coeff(j));
                    argRmin = j;
                }
            }
            element = (*calcElementsVector)[(*solverToOriginal)[argRmin] / 3];
            cout << "Direc,RMin,elementID,XYZ,Resistivity,preR:" << i << " " << rmin << " " << element->ID << " " << (*solverToOriginal)[argRmin] % 3 << " " << element->resistivity
                <<" "<<abs(r0norm[i].coeff(argRmin)) << endl;

            r0norm[i] = r[i];*/
            iters[i]++;

        }


            
    }
    m_error = 0.0;
    m_lastRelativeSolChange = 0.0;
    m_iters = 0;
    for (int i = 0; i < numOfSols; i++) {
        
        if (m_lastRelativeSolChangeVector[i] > m_lastRelativeSolChange) {
            m_lastRelativeSolChange = m_lastRelativeSolChangeVector[i];
        }
        if (iters[i] > m_iters) {
            m_iters = iters[i];
        }
        if (std::pow(r[i].squaredNorm() / (*rhs)[i].squaredNorm(), 0.5) > m_error) {
            m_error = std::pow(r[i].squaredNorm() / (*rhs)[i].squaredNorm(), 0.5);
        }
        rReturn[i] = r[i];
        //(*x)[i] = precond.RecoverSolution((*x)[i]);
    }

    outputfile.close();

    if (m_iters == maxIters) {
        return false;
    }
    else {
        return true;
    }
    //return;
}