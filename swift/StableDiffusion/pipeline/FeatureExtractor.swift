//
//  File.swift
//  
//
//  Created by Jingwen Zhu on 7/14/23.
//

import Foundation
import CoreML

/// A encoder model which produces latent samples from RGB images
@available(iOS 16.2, macOS 13.1, *)
public struct FeatureExtractor: ResourceManaging {
    
    public enum Error: String, Swift.Error {
        case sampleInputShapeNotCorrect
    }
    
    /// clip image encoder model + post math and adding noise from schedular
    var model: ManagedMLModel
    
    /// Create feature extractor from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled clip image encoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: An encoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }
    
    /// Ensure the model has been loaded into memory
    public func loadResources() throws {
        try model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
       model.unloadResources()
    }
    
    /// Prediction queue
    let queue = DispatchQueue(label: "featureextractor.predict")

    /// Encode image into latent embedding
    ///
    ///  - Parameters:
    ///    - image: Input image
    ///  - Returns: The extracted features from the image as MLShapedArray
    public func encode(_ image: CGImage) throws -> MLShapedArray<Float32> {
        let imageData = try image.plannerRGBShapedArray(minValue: 0.0, maxValue: 1.0)
        guard imageData.shape == inputShape else {
            // TODO: Consider auto resizing and croping similar to how Vision or CoreML auto-generated Swift code can accomplish with `MLFeatureValue`
            throw Error.sampleInputShapeNotCorrect
        }
        let dict = [inputName: MLMultiArray(imageData)]
        let input = try MLDictionaryFeatureProvider(dictionary: dict)
        print("feature extractor inputShape: \(inputShape)")
        print("feature extractor image shape: \(imageData.shape)")
        let result = try model.perform { model in
            try model.prediction(from: input)
        }
        print("feature extractor  result.featureNames : \(result.featureNames)")
        let outputName = result.featureNames.first!
        let outputValue = result.featureValue(for: outputName)!.multiArrayValue!
        let output = MLShapedArray<Float32>(outputValue)
        print("feature extractor  output shape : \(output.shape)")
        return output
    }
    
    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName["z"]!
        }
    }
    
    var inputName: String {
        inputDescription.name
    }
    
    /// The expected shape of the models latent sample input
    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}
