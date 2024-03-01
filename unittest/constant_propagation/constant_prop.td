Passes<list> {
    FrontendPass<pass> {
        src: fx_plus.compiler.passes;
    },
    Reassociation<pass> {
        src: fx_plus.compiler.passes;
    }
}

Test1<Model> {
    inputs<list> {
        x<tensor> {
            distrib: randn;
            size: (16, 64);
            dtype: float;
        },
        y<tensor> {
            distrib: randn;
            size: (16, 64);
            dtype: float;
        }
    }
}