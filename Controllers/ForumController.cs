using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BurnAnalysisApp.Data;
using BurnAnalysisApp.Models;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using System;
using System.IO;

namespace BurnAnalysisApp.Controllers
{
    [Route("api/forum")]
    [ApiController]
    public class ForumController : ControllerBase
    {
        private readonly AppDbContext _context;

        public ForumController(AppDbContext context)
        {
            _context = context;
        }

        [HttpPost("addPost")]
        public async Task<IActionResult> AddPost([FromBody] ForumPost forumPost)
        {
            if (forumPost == null || forumPost.PatientID == 0)
            {
                return BadRequest("Geçersiz veri.");
            }

            try
            {
                // Veritabanında var olan hastayı tekrar eklemek yerine, mevcut kaydı takip etmesini sağlıyoruz
                var existingPatient = await _context.Patients.FindAsync(forumPost.PatientID);

                if (existingPatient == null)
                {
                    return NotFound("Hasta bulunamadı.");
                }

                forumPost.Patient = null; // Patient nesnesini null yaparak tekrar eklemeyi önlüyoruz

                _context.ForumPosts.Add(forumPost);
                await _context.SaveChangesAsync();

                return Ok(new { message = "Forum gönderisi başarıyla eklendi." });

            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Sunucu hatası: {ex.Message}");
            }
        }
        //yeni
        [HttpGet("getPost/{postId}")]
        public async Task<IActionResult> GetPostById(int postId)
        {
            var post = await _context.ForumPosts
                .Include(fp => fp.Patient)   // Hasta bilgilerini getir
                .Include(fp => fp.Comments)  // Yoruları getir
                .FirstOrDefaultAsync(fp => fp.ForumPostID == postId);

            if (post == null)
            {
                return NotFound("Forum gönderisi bulunamadı.");
            }

            var postViewModel = new
            {
                post.ForumPostID,
                post.PatientID,
                post.PhotoPath,
                Patient = post.Patient,
                Comments = post.Comments.Select(c => new
                {
                    c.CommentID,
                    c.Content,
                    c.CreatedAt,
                    c.DoctorName
                }),
                post.DoctorName
            };

            return Ok(postViewModel);
        }


        // 2️⃣ Forum gönderilerini getir
        [HttpGet("getAll")]
        public async Task<IActionResult> GetAllPosts()
        {
            var posts = await _context.ForumPosts
                .Include(fp => fp.Patient) // **Hasta bilgilerini de getir**
                .Include(fp => fp.Comments)
                .ToListAsync();

            var postViewModels = posts.Select(post => new
            {
                post.ForumPostID,
                post.PatientID,
                post.PhotoPath,
                Patient = post.Patient,
                Comments = post.Comments,
                DoctorName = post.DoctorName // Assuming ForumPost has DoctorName
            }).ToList();

            return Ok(postViewModels);
        }

        [HttpPost("addComment/{postId}")] //burada bildirim atmak demek yorum yapıldıgı an bildirim gidiyor demek
        public async Task<IActionResult> AddComment(int postId, [FromBody] Comment comment)
        {
            if (comment == null || string.IsNullOrEmpty(comment.Content))
            {
                return BadRequest("Yorum boş olamaz!");
            }

            var post = await _context.ForumPosts
                .Include(f => f.Comments)
                .FirstOrDefaultAsync(f => f.ForumPostID == postId);

            if (post == null)
            {
                return NotFound("Gönderi bulunamadı!");
            }

            //  **Yorumu kaydet**
            comment.ForumPostID = postId;
            _context.Comments.Add(comment);
            await _context.SaveChangesAsync();

            //  **Yorum Yapan Doktorun Adını Veritabanından Çek**
            var commentingDoctor = await _context.Doctors
                .FirstOrDefaultAsync(d => d.Name == comment.DoctorName);

            //  **Post Sahibi Doktoru Veritabanından Çek**
            var postOwnerDoctor = await _context.Doctors
                .FirstOrDefaultAsync(d => d.Name == post.DoctorName);

            if (postOwnerDoctor != null && commentingDoctor != null && postOwnerDoctor.Name != commentingDoctor.Name)//post sahibi yorum yaptıgında bildirim gitmesin !=
            {
                var notification = new Notification
                {
                    DoctorID = postOwnerDoctor.DoctorID,
                    Message = $"Dr. {commentingDoctor.Name}, gönderinize yorum yaptı: {comment.Content}",
                    IsRead = false,
                    CreatedAt = DateTime.UtcNow,
                    ForumPostID = post.ForumPostID
                };

                _context.Notifications.Add(notification);
                await _context.SaveChangesAsync();
            }

            return Ok(comment);
        }

        // 4️⃣ Forum postu ve yorumları sil (ID güncellendi)
        [HttpDelete("delete/{postId}")]
        public async Task<IActionResult> DeleteForumPost(int postId)
        {
            var post = await _context.ForumPosts
                .Include(f => f.Comments)
                .FirstOrDefaultAsync(f => f.ForumPostID == postId);

            if (post == null) return NotFound();

            _context.ForumPosts.Remove(post);
            _context.Comments.RemoveRange(post.Comments);
            await _context.SaveChangesAsync();

            return NoContent();
        }



        // 5️⃣ Yorum sil
        [HttpDelete("deleteComment/{commentId}")]
        public async Task<IActionResult> DeleteComment(int commentId)
        {
            var comment = await _context.Comments.FindAsync(commentId);
            if (comment == null) return NotFound();

            _context.Comments.Remove(comment);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        [HttpPut("updateComment/{commentId}")]
        public async Task<IActionResult> UpdateComment(int commentId, [FromBody] CommentUpdateDTO updatedComment)
        {
            if (updatedComment == null)
            {
                return BadRequest("Güncellenmiş yorum verisi gönderilmedi!");
            }

            if (string.IsNullOrEmpty(updatedComment.Content))
            {
                return BadRequest("Yorum içeriği boş olamaz!");
            }

            var comment = await _context.Comments.FindAsync(commentId);
            if (comment == null)
            {
                return NotFound("Yorum bulunamadı!");
            }

            comment.Content = updatedComment.Content; // Yeni içeriği güncelle
            await _context.SaveChangesAsync();

            return Ok(comment);
        }
        [HttpPost("addVoiceRecording/{postId}")]
        public async Task<IActionResult> AddVoiceRecording(int postId, [FromForm] IFormFile file, [FromForm] string? doctorName)
        {
            if (file == null || file.Length == 0)
                return BadRequest("Ses dosyası yüklenmedi.");

            var post = await _context.ForumPosts.FindAsync(postId);
            if (post == null)
                return NotFound("Forum gönderisi bulunamadı.");

            try
            {
                // Eğer doktor adı null veya boş ise, varsayılan olarak bilinmeyen doktor yaz
                if (string.IsNullOrEmpty(doctorName))
                {
                    doctorName = "Bilinmeyen Doktor";
                    // Burada bir loglama yapabilir veya kullanıcıya bir uyarı mesajı gönderebilirsiniz
                    Console.WriteLine("Doktor adı alınamadı, varsayılan değer kullanıldı.");
                }

                // Dosya adını ve kaydedileceği yolu oluştur
                var fileName = $"{postId}_{Guid.NewGuid()}{Path.GetExtension(file.FileName)}";
                var uploadsFolder = Path.Combine("wwwroot", "uploads", "Audio");

                // Eğer "uploads/Audio" klasörü yoksa oluştur
                if (!Directory.Exists(uploadsFolder))
                {
                    Directory.CreateDirectory(uploadsFolder);
                }

                var filePath = Path.Combine(uploadsFolder, fileName);
                var relativePath = Path.Combine("uploads", "Audio", fileName); // Veritabanına kaydedilecek

                // Dosyayı kaydet
                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }

                // Ses kaydını oluştur ve veritabanına ekle
                var voiceRecording = new VoiceRecording
                {
                    ForumPostID = postId,
                    DoctorName = doctorName,
                    FilePath = relativePath,
                    CreatedAt = DateTime.UtcNow
                };

                _context.VoiceRecordings.Add(voiceRecording);
                await _context.SaveChangesAsync();

                return Ok(new { message = "Ses kaydı başarıyla yüklendi.", filePath = relativePath });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Sunucu hatası: {ex.Message}");
            }
        }
        [HttpGet("getVoiceRecordings/{postId}")]
        public async Task<IActionResult> GetVoiceRecordings(int postId)
        {
            var recordings = await _context.VoiceRecordings
                .Where(vr => vr.ForumPostID == postId)
                .ToListAsync();

            if (recordings == null || recordings.Count == 0)
            {
                return NotFound("Ses kaydı bulunamadı.");
            }

            return Ok(recordings);
        }

        [HttpDelete("deleteVoiceRecording/{recordingId}")]
        public async Task<IActionResult> DeleteVoiceRecording(int recordingId)
        {
            var recording = await _context.VoiceRecordings.FindAsync(recordingId);
            if (recording == null)
            {
                return NotFound("Ses kaydı bulunamadı.");
            }

            try
            {
                // Dosya yolunu al
                var filePath = Path.Combine("wwwroot", recording.FilePath);

                // Dosya varsa sil
                if (System.IO.File.Exists(filePath))
                {
                    System.IO.File.Delete(filePath);
                }

                // Veritabanından kaydı sil
                _context.VoiceRecordings.Remove(recording);
                await _context.SaveChangesAsync();

                return Ok(new { message = "Ses kaydı başarıyla silindi." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Sunucu hatası: {ex.Message}");
            }
        }
    }
}