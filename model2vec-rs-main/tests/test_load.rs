use approx::assert_relative_eq;
use model2vec_rs::model::StaticModel;

fn encode_with_model(path: &str) -> Vec<f32> {
    // Helper function to load the model and encode "hello world"
    let model = StaticModel::from_pretrained(path, None, None, None)
        .unwrap_or_else(|e| panic!("Failed to load model at {path}: {e}"));

    let out = model.encode(&["hello world".to_string()]);
    assert_eq!(out.len(), 1);
    out.into_iter().next().unwrap()
}

#[test]
fn quantized_models_match_float32() {
    // Compare quantized models against the float32 model
    let base = "tests/fixtures/test-model-float32";
    let ref_emb = encode_with_model(base);

    for quant in &["float16", "int8"] {
        let path = format!("tests/fixtures/test-model-{}", quant);
        let emb = encode_with_model(&path);

        assert_eq!(emb.len(), ref_emb.len());

        for (a, b) in ref_emb.iter().zip(emb.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-1);
        }
    }
}
