{
c1 = new TCanvas("c1","FIT",200,10,700,500);
c1->SetGrid();
f= new TF1("f1","[0]/(x^2)+[1]",0,5);
f->SetParNames("Multiplicative","Addendum");
f->SetParameters(100.,0.);
gr = new TGraphErrors("./dosevsdistance.txt","%lg %lg %lg %lg" );
gr->Fit(f);
gr->SetTitle("Verifica andamento 1/r^2; Tempo [ms]; Dose [muGy] ");
gr->SetMarkerColor(4);
gr->SetMarkerStyle(21);
gr->Draw("AP");
c1->Update();
}
