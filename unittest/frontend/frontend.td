Test1<Model> {
    inputs<list> {
        input<tensor> {
            distrib: randn;
            size: (128, 64);
            dtype: float;
        }
    }
}