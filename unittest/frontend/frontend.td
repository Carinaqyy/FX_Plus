Passes<list> {
    FrontendPass<pass> {
        src: fx_plus.compiler.passes;
    }
}

Test1<Model> {
    inputs<list> {
        input<tensor> {
            distrib: randn;
            size: (128, 64);
            dtype: float;
        }
    }
}

Test2<Model> {
    inputs<list> {
        x<tensor> {
            distrib: randn;
            size: (256, 64);
            dtype: float;
        },
        y<tensor> {
            distrib: randn;
            size: (256, 64);
            dtype: float;
        }
    }
}

Test3<Model> {
    inputs<list> {
        x<tensor> {
            distrib: randn;
            size: (16, 64);
            dtype: float;
        }
    }
}