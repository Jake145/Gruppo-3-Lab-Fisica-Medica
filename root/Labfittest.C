{
c1 = new TCanvas("c1","FIT",200,10,700,500);
c1->SetGrid();
f= new TF1("f1","[0]/(x^2)+[1]",0,5);
f->SetParNames("A","B ");
f->SetParameters(100.,0.);
gr = new TGraphErrors("./data.txt","%lg %lg %lg %lg" );
gr->Fit(f);
gr->SetTitle("Fit Title; X [u.a.]; Y [u.a.] ");
gr->SetMarkerColor(4);
gr->SetMarkerStyle(21);
gr->Draw("AP");
c1->Update();
}
