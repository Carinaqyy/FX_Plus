Passes<list> {
    DecompositionPass<pass> {
        src: fx_plus.compiler.passes;
    }
}

TestAddmm<Model> {
    inputs<list> {
        lhs<tensor> {
            distrib: randn;
            size: (16, 64);
            dtype: float;
        },
        rhs<tensor> {
            distrib: randn;
            size: (64, 16);
            dtype: float;
        },
        bias<tensor> {
            distrib: randn;
            size: (16,);
            dtype: float;
        }
    }
}