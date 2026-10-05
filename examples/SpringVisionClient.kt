// Example for Spring Boot 3.2+ with spring-web. No Python dependency in the JVM app.
package example

import com.fasterxml.jackson.annotation.JsonProperty
import org.springframework.core.io.ByteArrayResource
import org.springframework.http.MediaType
import org.springframework.util.LinkedMultiValueMap
import org.springframework.web.client.RestClient

data class BoundingBox(val x1: Double, val y1: Double, val x2: Double, val y2: Double)
data class Detection(
    @JsonProperty("class_id") val classId: Int,
    val label: String,
    val confidence: Double,
    val bbox: BoundingBox,
)
data class DetectionResult(
    val model: String,
    val width: Int,
    val height: Int,
    @JsonProperty("inference_ms") val inferenceMs: Double,
    val detections: List<Detection>,
)

class SpringVisionClient(private val restClient: RestClient) {
    // Build the injected RestClient with baseUrl("http://localhost:8000").
    fun detect(imageBytes: ByteArray, filename: String): DetectionResult {
        val image = object : ByteArrayResource(imageBytes) {
            override fun getFilename(): String = filename
        }
        val multipart = LinkedMultiValueMap<String, Any>().apply { add("image", image) }
        return requireNotNull(
            restClient.post()
                .uri("/vision/detect?confidence=0.25")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(multipart)
                .retrieve()
                .body(DetectionResult::class.java)
        )
    }
}
