//
//  File.swift
//  
//
//  Created by Jingwen Zhu on 7/14/23.
//

import Foundation
import CoreML

/// U-Net noise prediction model for stable diffusion
@available(iOS 16.2, macOS 13.1, *)
public struct PostFuser: ResourceManaging {

    /// Model used to fuse and project text embeddings and image features into fuse embeddings
    ///
    /// It can be in the form of a single model or multiple stages
    var model: ManagedMLModel

    /// Creates a post fuse model
    ///
    /// - Parameters:
    ///   - url: Location of post fuse model  compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: model that will lazily load its required resources when needed or requested
    public init(modelAt url: URL,
                configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    /// Load resources.
    public func loadResources() throws {
        try model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        model.unloadResources()
    }
    
    /// Prediction queue
    let queue = DispatchQueue(label: "postfuser.predict")

    /// fuse and project text embeddings and image features into fused embeddings
    ///
    /// - Parameters:
    ///   - textEmbeddings: text embeddings from text encoder
    ///   - imageEmbeddings: image embeddings from feature extract
    ///   - imageTokenMask: mask of which embedding in text embeddings to be concatenated with image embeddings
    ///   - numObjects: how many subjects will be generated
    /// - Returns: fused embeddings
    func fuse(
        textEmbeddings: MLShapedArray<Float32>,
        imageEmbeddings: MLShapedArray<Float32>,
        imageTokenMask: [Int32],
        numObjects: [Int32]
    ) throws -> MLShapedArray<Float32> {

//        let numObjects = MLShapedArray<Int32>(scalars:[Int32(1)],shape:[1])
        let imageTokenMaskArray = MLShapedArray<Int32>(scalars: imageTokenMask, shape: [1, 77])
        let numObjectsArray = MLShapedArray<Int32>(scalars: numObjects, shape: [1])
        let dict = ["text_embeds": MLMultiArray(textEmbeddings),
                    "object_embeds": MLMultiArray(imageEmbeddings),
                    "image_token_mask": MLMultiArray(imageTokenMaskArray),
                    "num_objects": MLMultiArray(numObjectsArray)]
        let input = try MLDictionaryFeatureProvider(dictionary: dict)
        
        let result = try model.perform { model in
            try model.prediction(from: input)
        }
        
        let outputName = result.featureNames.first!
        let outputValue = result.featureValue(for: outputName)!.multiArrayValue!
        let output = MLShapedArray<Float32>(outputValue)
        return output
    }
    
    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.value
        }
    }

    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}
